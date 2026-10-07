"""
Regression tests for the bugs FSCT-Turbo has fixed.

    python -m unittest discover -s tests            # fast checks, about a minute
    set FSCT_E2E=1                                  # then the same command adds
    python -m unittest discover -s tests            # a full run of the example plot

The end-to-end test runs the whole pipeline on data/test/example.las twice, at
different batch sizes and core counts, and requires bit-identical labels and
the exact stem-point count, which pins segmentation down.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(ROOT, "scripts"), ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fsct_job  # noqa: E402


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()

EXAMPLE = os.path.join(ROOT, "data", "test", "example.las")


class BoxExtraction(unittest.TestCase):
    """The k-d tree box gather must equal the original full scan, order included."""

    def test_tree_gather_matches_full_scan(self):
        from scipy import spatial
        from preprocessing import Preprocessing

        rng = np.random.default_rng(1)
        xyz = rng.uniform(0, 12, size=(60000, 3))
        cloud = np.column_stack([xyz, np.arange(len(xyz))])  # row index exposes any reordering
        box = np.array([6.0, 6.0, 6.0])
        centres = np.stack(np.meshgrid(*[np.linspace(0, 12, 5)] * 3), -1).reshape(-1, 3)
        ids = np.arange(len(centres))
        tree_dir, scan_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            kdtree = spatial.cKDTree(cloud[:, :3], leafsize=64)
            # max_points_per_box below the box size, so the seeded subsample runs too.
            Preprocessing.threaded_boxes(cloud, box, 1000, 5000, tree_dir + os.sep, ids, centres, kdtree, 0)
            Preprocessing.threaded_boxes(cloud, box, 1000, 5000, scan_dir + os.sep, ids, centres, None, 0)
            files = sorted(os.listdir(scan_dir))
            self.assertTrue(files)
            self.assertEqual(sorted(os.listdir(tree_dir)), files)
            for name in files:
                np.testing.assert_array_equal(
                    np.load(os.path.join(tree_dir, name)), np.load(os.path.join(scan_dir, name)), err_msg=name)
        finally:
            shutil.rmtree(tree_dir, ignore_errors=True)
            shutil.rmtree(scan_dir, ignore_errors=True)


def _batched_cloud(sizes, seed=1, extent=1.0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    pos = torch.rand(sum(sizes), 3, generator=generator) * extent
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    return pos, batch


class Network(unittest.TestCase):
    """A box's labels must depend only on the box: not device, batch or batch-mates."""

    def test_cpu_neighbours_match_the_cuda_kernel(self):
        # The CUDA kernel keeps, per query, the first max_num_neighbors points
        # within r in index order. This is that loop, written out.
        import torch
        from model import first_neighbours

        pos, batch = _batched_cloud([1500, 900], extent=0.6)  # dense: most queries exceed 64
        queries = torch.arange(0, pos.size(0), 7)
        row, col = first_neighbours(pos, pos[queries], 0.2, batch, batch[queries], 64)
        got = {}
        for q, c in zip(row.tolist(), col.tolist()):
            got.setdefault(q, []).append(c)
        overfull = 0
        for q, point in enumerate(queries.tolist()):
            same = (batch == batch[point]).nonzero().flatten()
            inside = same[((pos[same] - pos[point]) ** 2).sum(1) < 0.2 ** 2]
            overfull += len(inside) > 64
            self.assertEqual(got.get(q, []), inside[:64].tolist(), f"query {q}")
        self.assertGreater(overfull, len(queries) // 2, "test cloud is not dense enough to exercise the limit")

    def test_sampling_start_is_set_per_sample(self):
        import torch
        from model import seeded_fps

        pos, batch = _batched_cloud([1000, 1200, 800])
        u = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        together = seeded_fps(pos, batch, 0.1, u)
        self.assertTrue(torch.equal(together, seeded_fps(pos, batch, 0.1, u)))
        # The middle sample on its own samples the same points.
        alone = seeded_fps(pos[1000:2200], torch.zeros(1200, dtype=torch.long), 0.1, u[1:2])
        self.assertTrue(torch.equal(together[batch[together] == 1], alone + 1000))
        self.assertEqual(int(together[100]), 1000 + int(0.5 * 1200))  # starts at floor(u * n)
        self.assertFalse(torch.equal(together, seeded_fps(pos, batch, 0.1, u.flip(0))))

    def test_interpolation_matches_pyg_and_repeats(self):
        import torch
        from torch_geometric.nn import knn_interpolate
        from model import ordered_knn_interpolate

        pos_x, batch_x = _batched_cloud([200, 150], seed=2)
        pos_y, batch_y = _batched_cloud([900, 700], seed=3)
        x = torch.rand(350, 16, generator=torch.Generator().manual_seed(4))
        ours = ordered_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=3)
        torch.testing.assert_close(ours, knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=3))
        self.assertTrue(torch.equal(ours, ordered_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=3)))

    def test_self_loops_match_pyg_for_one_box(self):
        # For a single box, the per-sample self-loops must be exactly what
        # PointNetConv(add_self_loops=True) builds - the trained behaviour.
        import torch
        from torch_geometric.nn import PointNetConv
        from torch_geometric.utils import add_self_loops, remove_self_loops
        from model import first_neighbours, per_sample_self_loops, seeded_fps

        pos, batch = _batched_cloud([1200], extent=0.8)
        idx = seeded_fps(pos, batch, 0.1, torch.tensor([0.4], dtype=torch.float64))
        row, col = first_neighbours(pos, pos[idx], 0.2, batch, batch[idx], 64)
        edges = torch.stack([col, row])
        theirs, _ = add_self_loops(remove_self_loops(edges)[0], num_nodes=idx.numel())
        ours = per_sample_self_loops(col, row, batch, batch[idx])
        self.assertTrue(torch.equal(ours, theirs))
        # And the convolution agrees with PyG's own self-loop handling.
        torch.manual_seed(0)
        conv = PointNetConv(torch.nn.Linear(3, 8))
        expected = conv(None, (pos, pos[idx]), edges)
        conv.add_self_loops = False
        self.assertTrue(torch.equal(conv(None, (pos, pos[idx]), ours), expected))

    def test_labels_do_not_depend_on_batch_mates(self):
        import torch
        from torch_geometric.data import Batch, Data
        from model import Net

        torch.manual_seed(0)
        net = Net(num_classes=4).eval()
        samples = []
        for i, size in enumerate((2500, 3000)):
            pos, _ = _batched_cloud([size], seed=10 + i, extent=3.0)
            samples.append(Data(pos=pos, x=None, fps_u=torch.tensor([[0.3 + 0.2 * i, 0.7]], dtype=torch.float64)))
        with torch.no_grad():
            together = net(Batch.from_data_list(samples)).squeeze(0)
            alone = torch.cat([net(Batch.from_data_list([s])).squeeze(0) for s in samples], dim=1)
            again = net(Batch.from_data_list(samples)).squeeze(0)
        self.assertTrue(torch.equal(together, again), "a rerun must repeat exactly")
        # Scores, not just labels: an untrained network's argmax hides most
        # differences. Tolerance covers matmul rounding at a different batch
        # shape; a box leaking into its batch-mate moves scores far more.
        torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-4)

    def test_dataset_reads_boxes_in_id_order_with_per_box_starts(self):
        from inference import TestingDataset

        work = tempfile.mkdtemp()
        try:
            for box_id in (12, 3, 7):
                np.save(os.path.join(work, f"{box_id:07d}.npy"), np.random.default_rng(box_id).random((50, 3)))
            dataset = TestingDataset(work + os.sep, 20000, "cpu", random_seed=0)
            self.assertEqual([os.path.basename(f) for f in dataset.filenames],
                             ["0000003.npy", "0000007.npy", "0000012.npy"])
            starts = [dataset[i].fps_u for i in range(3)]
            self.assertTrue(all(s.shape == (1, 2) for s in starts))
            self.assertTrue(np.array_equal(dataset[0].fps_u.numpy(), starts[0].numpy()))
            self.assertFalse(np.array_equal(starts[0].numpy(), starts[1].numpy()))
        finally:
            shutil.rmtree(work, ignore_errors=True)


class Parameters(unittest.TestCase):
    def test_defaults_cover_every_run_parameter(self):
        # Every key upstream's scripts/run.py sets must have a default, or an
        # entry point that relies on the defaults fails with a KeyError.
        source = read("scripts", "run.py")
        block = source[source.index("parameters = dict("):source.index("parameters.update(other_parameters)")]
        run_keys = set(re.findall(r"^\s*(\w+)=", block, re.MULTILINE)) - {"point_cloud_filename"}
        self.assertEqual(run_keys - set(fsct_job.DEFAULT_PARAMETERS), set())
        self.assertIsNone(fsct_job.DEFAULT_PARAMETERS["plot_centre"])

    def test_job_parameters(self):
        from other_parameters import other_parameters

        p = fsct_job.job_parameters(batch_size=4)
        self.assertEqual(p["batch_size"], 4)
        self.assertEqual(p["random_seed"], other_parameters["random_seed"])
        with self.assertRaises(KeyError):
            fsct_job.job_parameters(batch_sise=4)

    def test_wrapper_config_is_valid(self):
        config = json.loads(read("wrapper_config.json"))
        fsct_job.job_parameters(**config["default_parameters"])  # raises on an unknown key

    def test_ui_summary_columns_are_written_by_fsct(self):
        # A UI once read columns FSCT never writes, and showed nothing.
        headers = read("scripts", "preprocessing.py")
        for ui in ("fsct_desktop.py",):
            source = read(ui)
            for column in ("Num Trees in Plot", "Stems/ha", "Mean DBH", "Mean Height", "Total Volume 1"):
                self.assertIn(f"'{column}'", source, f"{ui} does not show {column}")
                self.assertIn(f'"{column}"', headers)


class JobProcess(unittest.TestCase):
    def _child(self, code):
        return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def test_lines_marks_carriage_return_progress_as_transient(self):
        # Text mode, as FSCT prints: on Windows every "\n" arrives as "\r\n".
        job = self._child(r"import sys; sys.stdout.write('one\n 1 / 3\r 2 / 3\rtwo\nlast')")
        self.assertEqual(list(fsct_job.lines(job)),
                         [("one", False), (" 1 / 3", True), (" 2 / 3", True), ("two", False), ("last", False)])

    def test_lines_joins_crlf_split_across_reads(self):
        job = self._child("import sys, time; o = sys.stdout.buffer; o.write(b'a\\r'); o.flush(); "
                          "time.sleep(0.5); o.write(b'\\nb\\r\\n'); o.flush()")
        self.assertEqual(list(fsct_job.lines(job)), [("a", False), ("b", False)])

    def test_stage_for(self):
        self.assertEqual(fsct_job.stage_for("Starting multithreaded cylinder fitting..."),
                         "Measuring: cylinder fitting")
        self.assertIsNone(fsct_job.stage_for("Saved."))

    def test_stop_kills_grandchildren(self):
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil not installed")
        grandchild = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); print('ready', flush=True); time.sleep(120)"
        job = self._child(grandchild)
        self.assertEqual(job.stdout.readline().strip(), b"ready")
        tree = [job.pid] + [c.pid for c in psutil.Process(job.pid).children(recursive=True)]
        self.assertEqual(len(tree), 2)
        fsct_job.stop(job)
        job.stdout.close()
        self.assertIsNotNone(job.returncode)
        for pid in tree:
            self.assertFalse(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE,
                             f"process {pid} survived stop()")


class BatchProcess(unittest.TestCase):
    def test_failure_gives_nonzero_exit_and_config_is_found_from_anywhere(self):
        work = tempfile.mkdtemp()
        try:
            with open(os.path.join(work, "broken.las"), "wb") as f:
                f.write(b"not a point cloud")
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "batch_process.py"), work, "--no-recursive"],
                cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
            self.assertEqual(result.returncode, 1, result.stdout[-2000:])
            self.assertNotIn("not found; using the default parameters", result.stdout)
            self.assertNotIn("'plot_centre'", result.stdout)
        finally:
            shutil.rmtree(work, ignore_errors=True)


@unittest.skipUnless(os.environ.get("FSCT_E2E", "").strip() == "1", "set FSCT_E2E=1 to run the full pipeline")
class EndToEnd(unittest.TestCase):
    # Stem points on the example plot with the default settings. The GPU and the
    # CPU see identical neighbourhoods and samples, so they differ only by
    # floating-point rounding: 84 of 673,517 labels.
    STEM_POINTS = {"cuda": 176992, "cpu": 177002}

    def _run(self, work, name, **overrides):
        folder = os.path.join(work, name)
        os.makedirs(folder)
        src = os.path.join(folder, "example.las")
        shutil.copy(EXAMPLE, src)
        job = fsct_job.start(fsct_job.job_parameters(point_cloud_filename=src, **overrides))
        output = [line for line, transient in fsct_job.lines(job) if not transient]
        self.assertEqual(job.returncode, 0, "\n".join(output[-30:]))
        device = "cuda" if any("inference on device: cuda" in line for line in output) else "cpu"
        return fsct_job.output_dir_for(src), device

    def test_example_plot_is_exact_and_independent_of_batch_size_and_cores(self):
        import filecmp
        import laspy
        import pandas as pd

        work = tempfile.mkdtemp()
        try:
            a, device = self._run(work, "a", batch_size=2, num_cpu_cores=8)
            b, _ = self._run(work, "b", batch_size=4, num_cpu_cores=0)

            def labels(out):
                las = laspy.read(os.path.join(out, "segmented.las"))
                xyz = np.column_stack([las.X, las.Y, las.Z])
                return np.asarray(las["label"]).astype(int)[np.lexsort(xyz.T[::-1])]

            label_a = labels(a)
            np.testing.assert_array_equal(label_a, labels(b))
            self.assertEqual(int((label_a == 3).sum()), self.STEM_POINTS[device])  # stem class
            for name in ("tree_data.csv", "taper_data.csv", "cleaned_cyls.csv"):
                self.assertTrue(filecmp.cmp(os.path.join(a, name), os.path.join(b, name), shallow=False), name)
            self.assertEqual(len(pd.read_csv(os.path.join(a, "tree_data.csv"))), 4)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
