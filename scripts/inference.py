from abc import ABC
import torch
from torch_geometric.data import Dataset, Data

# DataLoader moved out of torch_geometric.data in PyG 2.0 and was removed
# from it in 2.1. Keep the old path working for anyone on an older PyG.
try:
    from torch_geometric.loader import DataLoader
except ImportError:  # PyG < 2.0
    from torch_geometric.data import DataLoader
import numpy as np
import glob
import pandas as pd
from model import Net
from sklearn.neighbors import NearestNeighbors
import os
import time
from tools import get_fsct_path
from tools import load_file, save_file
from fsct_exceptions import DataQualityError
import shutil
import sys

sys.setrecursionlimit(10**8)  # Can be necessary for dealing with large point clouds.


class TestingDataset(Dataset, ABC):
    def __init__(self, root_dir, points_per_box, device):
        super().__init__()
        self.filenames = glob.glob(root_dir + "*.npy")
        self.device = device
        self.points_per_box = points_per_box

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point_cloud = np.load(self.filenames[index])
        # Build the sample on the CPU and let the training loop move the whole
        # collated batch to the GPU in one go. Copying each sample separately
        # here meant one small host-to-device transfer per box, and CUDA
        # tensors in a Dataset also rule out pinned memory and worker
        # processes.
        pos = np.ascontiguousarray(point_cloud[:, :3], dtype=np.float32)
        pos = torch.from_numpy(pos)

        # Place sample at origin
        local_shift = torch.round(torch.mean(pos, dim=0))
        pos = pos - local_shift
        return Data(pos=pos, x=None, local_shift=local_shift)


def choose_most_confident_label(point_cloud, original_point_cloud, chunk_size=250000):
    """
    Args:
        original_point_cloud: The original point cloud to be labeled.
        point_cloud: The segmented point cloud (often slightly downsampled from the process).
        chunk_size: Number of query points whose neighbours are held in memory
                    at once. Affects peak memory only, never the result.

    Returns:
        The original point cloud with segmentation labels added.
    """

    print("Choosing most confident labels...")
    # n_jobs=-1 puts the kNN query on every core. Without it sklearn runs the
    # query sequentially, which was 3.1 s of a 5.9 s call on the benchmark.
    neighbours = NearestNeighbors(
        n_neighbors=16, algorithm="kd_tree", metric="euclidean", radius=0.05, n_jobs=-1
    ).fit(point_cloud[:, :3])

    # Slice the four class-probability columns *before* the fancy index.
    # point_cloud[indices] built an (N, 16, 7) temporary - roughly 600 MB for a
    # 700k-point cloud - when only 4 of those 7 columns are ever read.
    class_probabilities = np.ascontiguousarray(point_cloud[:, -4:])

    # Query in blocks, and ask only for the neighbour indices.
    #
    # Two allocations here used to scale with the whole plot at once.
    # kneighbors was returning distances that are immediately discarded - a
    # second (N, 16) float64 array - and the median ran over one (N, 16, 4)
    # gather, which numpy's partition then needs a working copy of. At 5.1M
    # points that is 623 MiB of unused distances and about 2.5 GiB of gather,
    # and the run died with
    #
    #     numpy._core._exceptions._ArrayMemoryError: Unable to allocate
    #     623. MiB for an array with shape (5104837, 16) and data type float64
    #
    # on a 16 GB machine - after the GPU had already done all the segmentation
    # work. Each row's median is independent of every other row, so taking them
    # a block at a time gives identical labels while peak memory is set by
    # chunk_size rather than by the size of the plot.
    num_points = original_point_cloud.shape[0]
    class_medians = np.empty((num_points, 4), dtype=np.float64)
    for start in range(0, num_points, chunk_size):
        stop = min(start + chunk_size, num_points)
        indices = neighbours.kneighbors(original_point_cloud[start:stop, :3], return_distance=False)
        class_medians[start:stop] = np.median(class_probabilities[indices], axis=1)

    labels = np.argmax(class_medians, axis=1).astype(np.float64)
    del class_medians

    return np.hstack((original_point_cloud, labels[:, np.newaxis]))


class SemanticSegmentation:
    def __init__(self, parameters):
        self.sem_seg_start_time = time.time()
        self.parameters = parameters

        if not self.parameters["use_CPU_only"]:
            print("Is CUDA available?", torch.cuda.is_available())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        print("Performing inference on device:", self.device)
        if not torch.cuda.is_available():
            print("Please be aware that inference will be much slower on CPU. An Nvidia GPU is highly recommended.")
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.directory = os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/") + "/"
        self.filename = self.filename.split("/")[-1]
        self.output_dir = self.directory + self.filename[:-4] + "_FSCT_output/"
        self.working_dir = self.directory + self.filename[:-4] + "_FSCT_output/working_directory/"

        self.filename = "working_point_cloud.las"
        self.directory = self.output_dir
        self.plot_summary = pd.read_csv(self.output_dir + "plot_summary.csv", index_col=None)
        self.plot_centre = [[float(self.plot_summary["Plot Centre X"].iloc[0]), float(self.plot_summary["Plot Centre Y"].iloc[0])]]

    def inference(self):
        # The farthest-point-sampling in scripts/model.py picks a random
        # starting point (torch_cluster.fps defaults to random_start=True), so
        # the network is not deterministic on its own. Seeding here makes a
        # rerun on the same file reproduce its labels exactly.
        random_seed = self.parameters.get("random_seed")
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)

        test_dataset = TestingDataset(
            root_dir=self.working_dir, points_per_box=self.parameters["max_points_per_box"], device=self.device
        )

        # Samples are plain CPU tensors now, so they can be pinned for faster
        # (and asynchronous) transfer to the GPU.
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        model = Net(num_classes=4).to(self.device)
        model_path = get_fsct_path("model") + "/" + self.parameters["model_filename"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Segmentation model not found at {model_path}.\n"
                "The trained model.pth must be in the 'model' folder of the FSCT project."
            )

        # Always map to the device we are about to run on, and load with
        # weights_only=True - this is a plain state_dict, and it became the
        # torch.load default in PyTorch 2.6.
        model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True),
            strict=False,
        )

        model.eval()

        # Mixed precision. On the benchmark cloud (RTX 3050, 4 GB) this took
        # segmentation from 17.0 s to 8.4 s at batch_size 4 - and it halves
        # activation memory, which matters more than the arithmetic on small
        # cards: in fp32 this model needs 3.3 GB at batch_size 6 and starts
        # spilling to host memory, which made *larger* batches slower.
        #
        # It shifts about 0.17% of point labels, all of them points where the
        # top two class probabilities were near-tied. For scale, FSCT's own
        # unseeded randomness used to move 8.76% of labels between runs of the
        # same file. Set use_amp=False in other_parameters.py to disable.
        use_amp = bool(self.parameters.get("use_amp", True)) and self.device.type == "cuda"
        if use_amp:
            print("Using mixed precision (fp16). Set use_amp=False to disable.")

        # TF32 costs about 0.015% of labels and is a straight win on Ampere
        # and newer for the layers autocast leaves in fp32.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        num_boxes = test_dataset.__len__()
        with torch.no_grad():

            self.output_point_cloud = np.zeros((0, 3 + 4))
            output_list = []
            for i, data in enumerate(test_loader):
                print("\r" + str(i * self.parameters["batch_size"]) + "/" + str(num_boxes))
                data = data.to(self.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = model(data)
                # Back to fp32 before the softmax so the class probabilities
                # keep full precision regardless of the autocast setting.
                out = out.float().permute(2, 1, 0).squeeze()

                # Everything below stays on the GPU until a single transfer at
                # the end. The previous version called .cpu() seven times per
                # iteration - three here and four more inside a per-sample loop
                # - and each one is a blocking device sync. Those calls were
                # 5.1 s of a 24 s segmentation on the benchmark cloud.
                out = torch.softmax(out, dim=1)

                # Undo the per-sample origin shift for every point at once.
                # local_shift is collated as a flat (3 * batch_size,) vector,
                # so view it as (batch_size, 3) and index it by each point's
                # batch id.
                local_shift = data.local_shift.view(-1, 3)
                pos = data.pos + local_shift[data.batch]

                # One device-to-host copy per batch instead of three.
                output = torch.cat((pos, out), dim=1).cpu().numpy()

                # PyG collates samples contiguously and in order, so splitting
                # by batch id here and re-stacking at the end reproduces this
                # array exactly. Append it whole and skip the split.
                output_list.append(output)

            if not output_list:
                raise DataQualityError(
                    "Semantic segmentation produced no output.\n"
                    "No sample boxes were generated during preprocessing, which usually means the "
                    "point cloud is too small or too sparse (each box needs more than "
                    f"{self.parameters['min_points_per_box']} points)."
                )
            self.output_point_cloud = np.vstack(output_list)
            print("\r" + str(num_boxes) + "/" + str(num_boxes))
        # Free the large intermediates. These only exist if the loop ran, which
        # the guard above has already established.
        del out, pos, output, output_list
        original_point_cloud, headers = load_file(
            self.directory + self.filename, headers_of_interest=["x", "y", "z", "red", "green", "blue"]
        )
        original_point_cloud[:, :2] = original_point_cloud[:, :2] - self.plot_centre
        self.output = choose_most_confident_label(self.output_point_cloud, original_point_cloud)
        # The per-box network output is finished with once the labels have been
        # transferred, and it is the largest array still alive here - one row
        # per point of every overlapping box, so bigger than the cloud itself.
        # Holding it through the save doubled peak memory for nothing.
        self.output_point_cloud = None
        del original_point_cloud
        self.output = np.asarray(self.output, dtype="float64")
        self.output[:, :2] = self.output[:, :2] + self.plot_centre
        save_file(
            self.output_dir + "segmented.las",
            self.output,
            headers_of_interest=["x", "y", "z", "red", "green", "blue", "label"],
        )

        self.sem_seg_end_time = time.time()
        self.sem_seg_total_time = self.sem_seg_end_time - self.sem_seg_start_time
        self.plot_summary["Semantic Segmentation Time (s)"] = self.sem_seg_total_time
        self.plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Semantic segmentation took", self.sem_seg_total_time, "s")
        print("Semantic segmentation done")
        if self.parameters["delete_working_directory"]:
            shutil.rmtree(self.working_dir, ignore_errors=True)
