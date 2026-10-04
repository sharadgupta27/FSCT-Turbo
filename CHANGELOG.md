# Changelog

Versioning is semantic, counting the original FSCT
([SKrisanski/FSCT](https://github.com/SKrisanski/FSCT)) as version 0. MAJOR is
for changes that alter measurements or break an existing workflow, MINOR for new
capability that leaves results alone, PATCH for fixes with no effect on output.
The version lives in `version.py`; `FSCT-Turbo.bat version` prints it.

## 1.0.0

The first release of FSCT-Turbo. Everything below is relative to the original
FSCT at commit `68e2f1e`, run with the library-compatibility edits it needs on
current Python packages. The segmentation model, its trained weights and the
measurement method are unchanged and remain the original authors' work.

Measurements on the 673,517-point `data/test/example.las`, on a laptop with a
Ryzen 7 7735HS (16 threads), 16 GB RAM and an RTX 3050 Laptop GPU (4 GB).

### Exact, consistent results

The original FSCT is not repeatable: two runs of the same file disagree on 7 to
8% of point labels, and on a CPU-only machine it segments markedly worse than
on a GPU. FSCT-Turbo gives one answer for a file and seed: bit-identical across
runs, CPU core counts, batch sizes, filesystems and positions within a batch
job. Each source of variation, and what replaced it:

- **Box subsampling.** Boxes over `max_points_per_box` were subsampled with an
  unseeded shuffle. They are seeded by the box's index in the global box array,
  so the result does not depend on how boxes are shared among threads.
- **Farthest-point sampling** in the network started from a random point,
  drawn by torch_cluster from a different generator on each device (torch's
  CUDA generator on the GPU, C's `rand()` on the CPU, which no seed reaches),
  from one stream shared across the batch. The start is now drawn per box from
  `(random_seed, box id)`, the same on every device and at any batch size.
- **The network's neighbourhoods on CPU.** Each sampled point gathers at most 64
  neighbours. torch_cluster's CUDA kernel keeps the first 64 in index order, the
  neighbourhoods the model was trained on; its CPU kernel keeps the first 64 in
  k-d tree order, a different set for 211 of 215 queries in a dense test cloud.
  CPU inference therefore fed the model neighbourhoods it had never seen, which
  is why CPU segmentation was worse (2 trees instead of 4 on the example plot).
  The CPU path now selects exactly what the CUDA kernel selects.
- **Self-loops leaked between boxes.** PointNetConv's self-loop step pairs
  sampled point *i* with input point *i* by index across the whole batch, so a
  box's features leaked into its batch-mate's and the labels depended on the
  batch size (batch 4 against batch 2: 21,593 labels). The same pairing is now
  made within each box, exactly what PyG does for a box on its own.
- **Box order.** Box files were read in the order the filesystem lists them,
  which decided batching and the order of the assembled cloud. They are read in
  box-id order.
- **Feature interpolation** summed neighbours with GPU atomics, in whatever
  order they landed. It is summed in a fixed order.
- **RANSAC circle fits** were unseeded; they are seeded per stem cluster.
- **Worker results** were collected in completion order, so the row order of
  every downstream array depended on process scheduling; they are collected in
  task order.
- **Precision.** Segmentation runs in fp32 by default, with deterministic cuDNN
  settings and TF32 off. `use_amp=True` runs it in fp16 (with TF32), about 15%
  faster on that stage and half the activation memory, but not bit-exact: 2
  of 673,517 labels differed between identical runs.

All of it follows `random_seed` in `scripts/other_parameters.py` (default `0`);
`None` restores the original non-reproducible behaviour.

Verified on the example plot, in fp32, against a run at batch size 2 on 8
cores (176,992 stem points, 4 trees):

| Run                                            | Labels that differ (of 673,517) |
| ---------------------------------------------- | ------------------------------- |
| Batch size 1                                   | 0                               |
| Batch size 4                                   | 0 (21,593 before)               |
| 16 cores, second run in the same process       | 0                               |
| Complete default run in a fresh process        | 0                               |
| CPU instead of GPU                             | 84 (212,634 before)             |
| Original FSCT, against another original run    | 7 to 8%, about 50,000           |

A complete CPU run measured the same four trees as the GPU run, with identical
DBH, height, `Volume_2` and `CCI_at_BH`; `Volume_1` differed by at most
0.003 m³ (0.3%). Before, the CPU found 2 trees.

The end-to-end test (`FSCT_E2E=1`) checks two complete runs at different batch
sizes and core counts: identical labels and byte-identical `tree_data.csv`,
`taper_data.csv` and `cleaned_cyls.csv`.

What remains approximate is floating-point arithmetic across hardware: a CPU
and a GPU round sums differently, which can flip a point whose two best class
scores are within rounding of each other.

### Measurement corrections

These change reported values relative to the original FSCT.

- **`Volume_1` was wrong twice over.** The frustum volume between consecutive
  cylinders read `(1/3) pi h (r1² + r1 + r2² + r2)`, adding a length to an
  area, where a frustum is `(1/3) pi h (r1² + r1·r2 + r2²)`; and every radius
  was halved a second time on the way in. The errors partly cancelled, so the
  numbers looked plausible: tree 1 of the example plot read 1.61 m³, more than
  a solid cylinder of its own DBH over its full height, against 0.66 m³ now and
  an independently estimated `Volume_2` of 0.53 m³.
- **`CCI_at_BH` used every tree's stem points**, so a neighbouring stem in the
  0.8 to 1.2 r annulus counted as coverage of this tree's circumference. It uses
  the tree's own points.
- **Branch-to-parent interpolation matched the wrong parent.** A mean and a norm
  without an axis made "the closest point of the current branch" whatever row
  came first, and the smallest-angle candidate was looked up in a filtered
  array but taken from the unfiltered one.
- **The "VOLUME 2" line was missing** from `text_point_cloud.las` in the default
  (uncropped) mode.

Kept on purpose, so measurements stay comparable with published FSCT results:

- **Cylinder sorting emits one cylinder short**, as the original loop does.
- **CCI sectors are not evenly spaced**: the sector angles are built in degrees
  but used as radians. `fix_cci_sectors=True` corrects it; it is off by default
  because CCI decides which cylinders survive, so stem count, DBH and height
  move with it.
- **The skeleton-point recovery step never worked** (it wrote into a temporary
  and searched only among unassigned points). `assign_unassigned_skeleton_points`
  makes it work; off by default for the same reason.

### Speed

Interleaved runs of the original and FSCT-Turbo, 8 workers, batch size 2, each
ratio comparing runs minutes apart:

| Stage           |     Original | FSCT-Turbo (fp32, default) |   FSCT-Turbo (fp16) |
| --------------- | -----------: | -------------------------: | ------------------: |
| Preprocessing   |       3.68 s |              0.85 s (4.3x) |       0.88 s (4.2x) |
| Segmentation    |      21.17 s |            18.36 s (1.15x) |     15.47 s (1.37x) |
| Post-processing |       1.53 s |             1.32 s (1.16x) |      1.24 s (1.23x) |
| Measurement     |     304.07 s |            13.81 s (22.0x) |     12.87 s (23.6x) |
| **Total**       | **330.46 s** |         **34.35 s (9.6x)** | **30.46 s (10.8x)** |

These were timed before the consistency changes above, which add some index
arithmetic per batch and turn TF32 off in fp32. Timed afterwards, alternating
with the code before them, segmentation took 23.1 s against 22.0 s (three runs
each, ranges 20.9 to 25.4 s and 19.0 to 23.5 s): within this machine's noise.

- **Circle fitting** was 92% of the original run: `skimage.measure.ransac` with
  `min_samples` at 30% of the slice and 10,000 trials, a combination for which
  the early-stopping rule never fires. Trials are now formed, solved and scored
  as arrays (2.6x on its own), capped at 1,000 and fitted on at most 1,500
  points per slice (69x in all, 2,039 ms to 29.4 ms per fit). The fitted radius
  moves by a median of 0.7 mm, against 0.1 mm between two runs of the original
  fit. `circle_fit_trials=10000` and `circle_fit_max_points=0` restore the
  original cost.
- **Box extraction** scanned the whole cloud once per box. One shared k-d tree,
  queried with a Chebyshev ball (exactly an axis-aligned cube) and gathered in
  index order, gives byte-identical boxes; the GIL is released, so the threads
  run in parallel.
- **Segmentation output** left the GPU in seven blocking copies per batch, four
  of them in a per-sample loop; now one.
- **Quadratic algorithms removed.** Cylinder sorting, cleaning and volume
  summation rebuilt a spatial index and copied the array per cylinder; one index
  and an active mask give the same order and neighbourhoods. Arrays grown with
  `np.vstack` inside loops (DTM construction, tree data, cylinder interpolation,
  text annotations) are stacked once. The DTM was re-triangulated three times
  per tree; once now.
- **Full scans in loops.** Slice and plane cutting sort once and binary-search;
  cover fractions use one matrix product and threaded counting queries; the
  skeleton-to-stem match used a k-d tree leaf size that made it brute force.
- **One worker pool**, started during GPU segmentation when memory allows,
  replaces four pools spawned on demand (each spawn re-imports the scientific
  stack per worker on Windows). Tasks are submitted largest first through a
  sliding window, with no barrier between batches.
- **Batch size.** Bigger is not faster on a small card: in fp32 on 4 GB, batch
  2 took 17.0 s, batch 4 25.3 s and batch 6 54.2 s, as activations spilled to
  host memory. The desktop app suggests a batch size from the detected VRAM.

### Reliability

- **Large clouds.** A 5.1-million-point plot ran out of memory at the end of
  segmentation on a 16 GB machine (the label transfer built about 3.5 GB of
  temporaries). Labels are now transferred in blocks of 250,000 points; the
  plot completes in 156 s. `load_file` and `save_file` no longer copy the whole
  cloud once per column.
- **A dead worker no longer hangs the run.** If the OS killed a measurement
  worker, the run waited forever. It now stops within 30 s with an explanation.
- **Plots with no trees** complete with an empty `tree_data.csv`; degenerate
  plots (no DTM grid cell, zero hull area) no longer end in `ZeroDivisionError`.
- **Current libraries.** `np.percentile(interpolation=)` (numpy 2),
  `read_csv(delim_whitespace=)` (pandas 3) and the `DataLoader` and
  `PointConv` moves in PyG 2 each crashed the original.
- **Paths and files.** `get_fsct_path` worked only in a folder named exactly
  `FSCT`. `load_file` crashed and `save_file` silently wrote nothing for
  `.LAS` in upper case or `.laz`. Optional subsampling duplicated points across
  slice boundaries and collected slices in completion order.
- **Training tools.** `training_monitor.py` caught the wrong exception and read
  a misspelt path, so it redrew an empty figure forever.

### Interfaces and tooling

The original is run by editing the parameters at the top of `scripts/run.py`.

- **Desktop application** (`fsct_desktop.py`): header-only file inspection,
  conversion, resampling and LAStools viewing, every parameter with an
  explanation, a live console with the current stage, a results browser, and a
  **Stop** button that ends the run and its worker processes.
- **Browser UI** (`fsct_web.py`): the same pipeline in Streamlit, with an
  interactive 3D view, distributions, a stem map and a DBH-height scatter. A run
  keeps going while the page is used and shows its stage and output live.
- **Both run the pipeline in a child process** (`fsct_job.py`), from one set of
  default parameters, so Stop can end it and a closed window does not leave it
  running.
- **Installer and launcher** (`FSCT-Turbo.bat`): builds the conda environment
  with a matching PyTorch, PyTorch Geometric and torch-cluster, installs both
  interfaces and downloads LAStools, asking nothing.
- **Unattended batch processing** (`batch_process.py`): a directory tree from
  the command line, parameters from `wrapper_config.json`, summaries combined,
  and an exit code that reports failures.
- **Checks**: `test_installation.py` (`FSCT-Turbo.bat verify`) for the
  environment, and regression tests in `tests/`
  (`python -m unittest discover -s tests`; `FSCT_E2E=1` adds a full run of the
  example plot).
