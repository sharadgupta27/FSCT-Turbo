import numpy as np
import time
import pandas as pd
from scipy import spatial
import threading
from tools import load_file, save_file, make_folder_structure, subsample_point_cloud, low_resolution_hack_mode
import os


class Preprocessing:
    def __init__(self, parameters):
        self.preprocessing_time_start = time.time()
        self.parameters = parameters
        self.filename = self.parameters["point_cloud_filename"].replace("\\", "/")
        self.directory = os.path.dirname(os.path.realpath(self.filename)).replace("\\", "/") + "/"
        self.filename = self.filename.split("/")[-1]
        self.box_dimensions = np.array(self.parameters["box_dimensions"])
        self.box_overlap = np.array(self.parameters["box_overlap"])
        self.min_points_per_box = self.parameters["min_points_per_box"]
        self.max_points_per_box = self.parameters["max_points_per_box"]
        self.num_cpu_cores = parameters["num_cpu_cores"]

        self.output_dir, self.working_dir = make_folder_structure(self.directory + self.filename)

        self.point_cloud, headers, self.num_points_orig = load_file(
            filename=self.directory + self.filename,
            plot_centre=self.parameters["plot_centre"],
            plot_radius=self.parameters["plot_radius"],
            plot_radius_buffer=self.parameters["plot_radius_buffer"],
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
            return_num_points=True,
        )

        self.num_points_trimmed = self.point_cloud.shape[0]

        if self.parameters["plot_centre"] is None:
            mins = np.min(self.point_cloud[:, :2], axis=0)
            maxes = np.max(self.point_cloud[:, :2], axis=0)
            self.parameters["plot_centre"] = (mins + maxes) / 2

        if self.parameters["subsample"]:
            self.point_cloud = subsample_point_cloud(
                self.point_cloud, self.parameters["subsampling_min_spacing"], self.num_cpu_cores
            )

        self.num_points_subsampled = self.point_cloud.shape[0]

        save_file(
            self.output_dir + "working_point_cloud.las",
            self.point_cloud,
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
        )

        self.point_cloud = self.point_cloud[:, :3]  # Trims off unneeded dimensions if present.

        if self.parameters["low_resolution_point_cloud_hack_mode"]:
            self.point_cloud = low_resolution_hack_mode(
                self.point_cloud,
                self.parameters["low_resolution_point_cloud_hack_mode"],
                self.parameters["subsampling_min_spacing"],
                self.parameters["num_cpu_cores"],
            )

            save_file(self.output_dir + self.filename[:-4] + "_hack_mode_cloud.las", self.point_cloud)

        # Global shift the point cloud to avoid loss of precision during segmentation.
        self.point_cloud[:, :2] = self.point_cloud[:, :2] - self.parameters["plot_centre"]

    @staticmethod
    def threaded_boxes(
        point_cloud,
        box_size,
        min_points_per_box,
        max_points_per_box,
        path,
        box_ids,
        point_divisions,
        kdtree=None,
        random_seed=None,
    ):
        """
        Write one .npy sample per box that contains enough points.

        `box_ids` gives the index of each box in the caller's full box_centres
        array. It used to be a single `id_offset` counting the boxes handed to
        earlier threads, which made a box's id - and so its RNG seed - depend on
        how many threads the work was split across. See the seeding note below.

        `kdtree` is a scipy cKDTree over point_cloud[:, :3]. It is optional
        only so the old call signature keeps working; without it this falls
        back to scanning the whole cloud per box, which is O(boxes x points)
        and dominates preprocessing on large clouds.
        """
        box_size = np.asarray(box_size, dtype=float)
        box_centre_mins = point_divisions - 0.5 * box_size
        box_centre_maxes = point_divisions + 0.5 * box_size

        # An axis-aligned cube is exactly a Chebyshev (p=inf) ball, so one
        # tree query returns the candidates for a box. Use the largest half
        # extent when the box is not a cube and let the mask below discard the
        # extra candidates.
        query_radius = float(np.max(0.5 * box_size))

        pds = len(point_divisions)

        for i in range(pds):
            if kdtree is not None:
                # Sorted, so the box keeps the cloud's point order. The tree
                # returns indices in traversal order, and order is not
                # cosmetic: the segmentation model's radius search keeps the
                # first 64 neighbours by index, so the same points in another
                # order get different labels. Gathering unsorted labelled ~9%
                # more of the example plot as stem than the original code, a
                # systematic shift rather than run-to-run noise.
                candidate_idx = kdtree.query_ball_point(
                    point_divisions[i], r=query_radius, p=np.inf, return_sorted=True
                )
                if len(candidate_idx) <= min_points_per_box:
                    continue  # cannot pass the test below, skip the gather
                candidates = point_cloud[candidate_idx]
            else:
                candidates = point_cloud

            # Re-apply the exact half-open bounds. The tree query is inclusive
            # on both sides and may be over-sized; this, with the sorted gather
            # above, keeps the box byte-identical to the original full scan.
            xyz = candidates[:, :3]
            keep = np.all((xyz >= box_centre_mins[i]) & (xyz < box_centre_maxes[i]), axis=1)
            box = candidates[keep]

            if box.shape[0] > min_points_per_box:
                if box.shape[0] > max_points_per_box:
                    # Was two random.shuffle() passes over a Python list of
                    # every index in the box - allocating and shuffling
                    # hundreds of thousands of Python ints to keep 20,000.
                    #
                    # Seed per box, not per thread: the boxes are spread over
                    # worker threads, so a shared or per-thread generator would
                    # make the result depend on thread scheduling.
                    #
                    # The id must identify the box itself, not its position in
                    # one thread's share of the work. It was `id_offset + i`,
                    # counting boxes handed to earlier threads, so the same
                    # physical box was box 1 on a single core, box 125 on four
                    # and box 32 on sixteen - three different seeds, three
                    # different subsamples of any box over max_points_per_box.
                    # Measured on data/test/example_clean.las, 4 cores and 16
                    # cores disagreed on 74,217 of 673,517 point labels (11%),
                    # enough to change the plot from 3 trees to 4. random_seed
                    # exists to make runs repeatable, and that has to hold
                    # across machines with different core counts too.
                    box_id = int(box_ids[i])
                    rng = np.random.default_rng(None if random_seed is None else [random_seed, box_id])
                    chosen = rng.choice(box.shape[0], size=max_points_per_box, replace=False)
                    box = np.asarray(box[chosen, :], dtype="float64")

                np.save(path + str(int(box_ids[i])).zfill(7) + ".npy", box)
        return 1

    def preprocess_point_cloud(self):
        print("Pre-processing point cloud...")
        point_cloud = self.point_cloud  # [self.point_cloud[:,4]!=5]
        Xmax = np.max(point_cloud[:, 0])
        Xmin = np.min(point_cloud[:, 0])
        Ymax = np.max(point_cloud[:, 1])
        Ymin = np.min(point_cloud[:, 1])
        Zmax = np.max(point_cloud[:, 2])
        Zmin = np.min(point_cloud[:, 2])

        X_range = Xmax - Xmin
        Y_range = Ymax - Ymin
        Z_range = Zmax - Zmin

        num_boxes_x = int(np.ceil(X_range / self.box_dimensions[0]))
        num_boxes_y = int(np.ceil(Y_range / self.box_dimensions[1]))
        num_boxes_z = int(np.ceil(Z_range / self.box_dimensions[2]))

        x_vals = np.linspace(
            Xmin, Xmin + (num_boxes_x * self.box_dimensions[0]), int(num_boxes_x / (1 - self.box_overlap[0])) + 1
        )
        y_vals = np.linspace(
            Ymin, Ymin + (num_boxes_y * self.box_dimensions[1]), int(num_boxes_y / (1 - self.box_overlap[1])) + 1
        )
        z_vals = np.linspace(
            Zmin, Zmin + (num_boxes_z * self.box_dimensions[2]), int(num_boxes_z / (1 - self.box_overlap[2])) + 1
        )

        box_centres = np.vstack(np.meshgrid(x_vals, y_vals, z_vals)).reshape(3, -1).T

        # Deal the box centres out to the worker threads round-robin. Same
        # assignment as before, without stepping through the array one row at
        # a time.
        point_divisions = [box_centres[i :: self.num_cpu_cores] for i in range(self.num_cpu_cores)]

        # One spatial index over the whole cloud, shared by every thread.
        # Each box then costs a tree query plus a scan of its own candidates,
        # instead of a boolean scan of all N points - the old version was
        # O(boxes x points). cKDTree queries release the GIL, so the threads
        # below now actually run concurrently.
        kdtree = spatial.cKDTree(self.point_cloud[:, :3], leafsize=64)

        # The global index of each box centre, split the same way the centres
        # themselves are. box_centres[thread::cores][i] is box_centres[thread +
        # i*cores], so these ids name the box regardless of the core count and
        # agree with a single-threaded run.
        box_id_divisions = [np.arange(len(box_centres))[i :: self.num_cpu_cores] for i in range(self.num_cpu_cores)]

        threads = []
        for thread in range(self.num_cpu_cores):
            t = threading.Thread(
                target=Preprocessing.threaded_boxes,
                args=(
                    self.point_cloud,
                    self.box_dimensions,
                    self.min_points_per_box,
                    self.max_points_per_box,
                    self.working_dir,
                    box_id_divisions[thread],
                    point_divisions[thread],
                    kdtree,
                    self.parameters.get("random_seed"),
                ),
            )
            threads.append(t)

        for x in threads:
            x.start()

        for x in threads:
            x.join()

        self.preprocessing_time_end = time.time()
        self.preprocessing_time_total = self.preprocessing_time_end - self.preprocessing_time_start
        print("Preprocessing took", self.preprocessing_time_total, "s")
        plot_summary_headers = [
            "PlotId",
            "Point Cloud Filename",
            "Plot Centre X",
            "Plot Centre Y",
            "Plot Radius",
            "Plot Radius Buffer",
            "Plot Area",
            "Num Trees in Plot",
            "Stems/ha",
            "Mean DBH",
            "Median DBH",
            "Min DBH",
            "Max DBH",
            "Mean Height",
            "Median Height",
            "Min Height",
            "Max Height",
            "Mean Volume 1",
            "Median Volume 1",
            "Min Volume 1",
            "Max Volume 1",
            "Total Volume 1",
            "Mean Volume 2",
            "Median Volume 2",
            "Min Volume 2",
            "Max Volume 2",
            "Total Volume 2",
            "Avg Gradient",
            "Avg Gradient X",
            "Avg Gradient Y",
            "Canopy Cover Fraction",
            "Understory Veg Coverage Fraction",
            "CWD Coverage Fraction",
            "Num Points Original PC",
            "Num Points Trimmed PC",
            "Num Points Subsampled PC",
            "Num Terrain Points",
            "Num Vegetation Points",
            "Num CWD Points",
            "Num Stem Points",
            "Preprocessing Time (s)",
            "Semantic Segmentation Time (s)",
            "Post processing time (s)",
            "Measurement Time (s)",
            "Total Run Time (s)",
        ]

        plot_summary = pd.DataFrame(np.zeros((1, len(plot_summary_headers))), columns=plot_summary_headers)

        # These two hold strings. Widen them before assigning, otherwise pandas
        # emits a "setting an item of incompatible dtype" FutureWarning on 2.x
        # and raises on 3.0, since the frame above is all float64.
        plot_summary = plot_summary.astype({"PlotId": object, "Point Cloud Filename": object})

        plot_summary["Preprocessing Time (s)"] = self.preprocessing_time_total
        plot_summary["PlotId"] = self.filename[:-4]
        plot_summary["Point Cloud Filename"] = self.parameters["point_cloud_filename"]
        plot_summary["Plot Centre X"] = self.parameters["plot_centre"][0]
        plot_summary["Plot Centre Y"] = self.parameters["plot_centre"][1]
        plot_summary["Plot Radius"] = self.parameters["plot_radius"]
        plot_summary["Plot Radius Buffer"] = self.parameters["plot_radius_buffer"]
        plot_summary["Num Points Original PC"] = self.num_points_orig
        plot_summary["Num Points Trimmed PC"] = self.num_points_trimmed
        plot_summary["Num Points Subsampled PC"] = self.num_points_subsampled

        plot_summary.to_csv(self.output_dir + "plot_summary.csv", index=False)
        print("Preprocessing done\n")
