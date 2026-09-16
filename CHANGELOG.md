# Changelog

Versioning is semantic: MAJOR for changes that alter measurements or break an
existing workflow, MINOR for new capability that leaves results alone, PATCH for
fixes with no effect on output. The version itself lives in `version.py`; run
`FSCT.bat version` to print it.

## 1.0.0

First versioned release.

This project is upstream FSCT — <https://github.com/SKrisanski/FSCT>, by Sean
Krisanski — with two user interfaces, an unattended installer and a
substantially faster, reproducible processing chain wrapped around it. The
segmentation model and the measurement method are the original work; what
follows is what differs.

Benchmarks below are on the 673k-point `data/test/example.las`, an RTX 3050
Laptop (4 GB) with 16 CPU cores, on an otherwise idle machine. Absolute
wall-clock times move a lot with background load — the same measurement stage
ran 19.5 s idle and 44 s with a browser and an editor open. Every before/after
pair was measured back to back under the same conditions, so the ratios hold
where the absolute numbers do not.

---

### Functionality added over upstream FSCT

Upstream is run by editing parameters at the top of `scripts/run.py` and
executing it, against a conda environment you assemble yourself.

- **Desktop application** (`fsct_desktop.py`) — file picker, header inspection
  without loading the points, parameter controls, live console, results browser
  and a plot summary view.
- **Browser UI** (`fsct_web.py`) — the same pipeline as a Streamlit app, with
  interactive 3D point cloud, height and DBH distributions, stem map and
  DBH-height scatter (`visualization_utils.py`).
- **One-shot installation** (`FSCT.bat`) — builds the conda environment,
  installs a matching PyTorch/PyG/torch-cluster triple, installs both
  interfaces and downloads LAStools, asking nothing. Also exposes `gui`, `web`,
  `setup`, `verify`, `lastools`, `version` and `help` subcommands.
- **Unattended batch processing** (`batch_process.py`) — process a directory
  tree of point clouds from the command line and combine the per-plot summaries,
  where upstream's directory mode opens a Tk folder dialog.
- **LAStools integration** (`setup_lastools.py`) — automatic download and
  configuration, with LAS/LAZ conversion, resampling and an external 3D viewer
  wired into both UIs.
- **Installation self-test** (`test_installation.py`, `FSCT.bat verify`) —
  checks the Python version, the deep-learning stack, the linear-algebra
  routines, point-cloud and reporting libraries, core files, the FSCT imports
  and optional GPU/LAStools/UI components.
- **Reproducible runs.** See below.
- **Versioning.** `version.py` as the single source of truth, surfaced in both
  UIs, the installation test, `FSCT.bat version` and `batch_process --version`.

### Reproducibility

Upstream runs are not repeatable: the same file processed twice disagrees on
roughly 10% of point labels. Four things were unseeded —

- boxes over `max_points_per_box` were subsampled with an unseeded
  `random.shuffle`
- the farthest-point-sampling in `scripts/model.py` picks a random start
  (`torch_cluster.fps` defaults to `random_start=True`)
- the RANSAC circle fits
- worker results were collected in completion order, so the row order of every
  downstream array depended on process scheduling

All four are now driven by `random_seed` in `scripts/other_parameters.py`
(default `0`); set it to `None` for the old behaviour. The box seed is derived
per box id and the circle-fit seed per stem cluster, so neither depends on
thread or worker scheduling — **results do not depend on the CPU core count**.

Two caveats: reproducibility holds for the same file, seed *and* batch size, as
changing the batch regroups the samples; and with `use_amp=True` reproduction is
very close but not bit-exact, measured at 2 differing labels out of 673,517
(0.0003%). Seeded fp32 runs matched exactly.

### Speed

Whole pipeline:

| Stage | Upstream | Here |
|---|---|---|
| Preprocessing | 4.04 s | 0.70 s |
| Segmentation | 24.1 s | 17.0 s |
| Post-processing | — | 1.6 s |
| Measurement | see below | 19.5 s |

The measurement stage was dominated by circle fitting. `skimage.measure.ransac`
was called with `min_samples` at 30% of the slice and `max_trials=10000` — a
combination for which RANSAC's early-stopping rule almost never fires, so nearly
every circle ran all 10,000 trials, each a Python-level `np.linalg.lstsq` plus a
residual pass over every point in the slice.

| Measurement | Upstream | Here | |
|---|---|---|---|
| One circle fit (60 real stem slices) | 2737 ms | 28.6 ms | **96x** |
| Cylinder fitting, all clusters, single process | 1739 ms/cyl | 29 ms/cyl | **60x** |

What changed:

- **Batched circle RANSAC.** Trials are sampled, fitted and scored with array
  operations instead of one Python loop iteration each, with the same dynamic
  stopping rule applied after every batch — same model, same scoring, same
  criterion. Agreement with skimage on 60 real stem slices: median radius
  difference 1.5 mm, 95th percentile 33 mm, against 35 mm of disagreement
  between two independent skimage runs on the same slices. `circle_fit_trials`
  and `circle_fit_max_points` control the trial count and point cap.
- **Indexed plane slicing.** Selecting points near each circle's plane scanned
  the whole stem cluster once per skeleton position; it now binary-searches a
  pre-sorted coordinate and returns an identical slice.
- **Cylinder sorting and cleaning are no longer quadratic.** Both rebuilt a
  spatial index and copied the whole array every iteration — O(n² log n) in
  cylinders. One index plus an active mask gives the same processing order and
  the same neighbourhoods.
- **Vectorised cylinder visualisation.** It dispatched one pool task per
  cylinder to produce 15 points; the two visualisation passes were 21 s of a
  55 s stage and are now under a second.
- **Parallel slice clustering**, with slice cutting binary-searching a sorted
  height instead of boolean-scanning the whole stem cloud per slice.
- **One shared worker pool.** Spawning a pool on Windows re-imports numpy,
  scipy, sklearn and hdbscan in every worker; that startup was being paid three
  times per run (slice clustering, cylinder fitting, cylinder cleaning).
- **Box extraction was O(boxes × points)** — it boolean-scanned the entire cloud
  once per box. Now one shared `cKDTree` queried with a Chebyshev (p=inf) ball,
  which is exactly an axis-aligned cube. `cKDTree` releases the GIL, so the
  worker threads genuinely run in parallel.
- **Segmentation output assembly** called `.cpu()` seven times per iteration,
  four inside a per-sample inner loop, each a blocking device sync. It also
  built an `(N, 16, 7)` temporary — about 600 MB at 700k points — to read four
  columns.
- **Mixed precision** (`use_amp`), which on a small GPU matters as much for
  memory as for arithmetic. Roughly halves both runtime and activation memory
  and shifts ~0.17% of labels; ignored on CPU.
- **DTM construction** grew its array with `np.vstack` inside a nested loop,
  making it quadratic in grid cells — 40,000 cells for a 100 × 100 m plot at
  0.5 m resolution.

The desktop app picks a default batch size from detected VRAM, because bigger is
not faster: in fp32 on a 4 GB card, batch 2 took 17.0 s at 1.37 GB peak, batch 4
took 25.3 s at 2.42 GB, and batch 6 took 54.2 s at 3.28 GB. Past roughly 2.5 GB
the driver pages activations to host memory and throughput collapses.

### Correctness and compatibility over upstream

Fixed:

- **Runs on current libraries.** `np.percentile(..., interpolation=)` was
  removed in numpy 2.0 and crashed the measurement stage just before tree
  heights were computed; `pandas.read_csv(delim_whitespace=)` was removed in
  pandas 3.0; `DataLoader` moved out of `torch_geometric.data` in PyG 2.0.
- **`get_fsct_path` worked only in a folder named exactly `FSCT`.** It truncated
  `os.getcwd()` at the first "FSCT" substring, so any other folder name resolved
  to a sibling directory that does not exist, and a path without "FSCT" in it
  raised `ValueError`. It is now derived from the module's own location.

Known upstream quirks, reproduced on purpose so measurements stay comparable:

- **Cylinder sorting emits one cylinder short.** The original loop stops with
  the last cylinder still unsorted and never emits it. The rewritten sort keeps
  that behaviour rather than silently changing every plot's cylinder count.
- **CCI sectors are not evenly spaced.** The sector angles are built in degrees
  but consumed as radians, so the 80 "evenly spaced sectors" are really 80
  arbitrary directions. They land quasi-uniformly, so CCI still roughly tracks
  circumferential coverage, but the values are not what the method describes.
  `fix_cci_sectors` in `other_parameters.py` corrects it and is **off by
  default**: CCI decides which cylinders survive, so stem count, DBH and tree
  height all move with it.

---

### Changes since the last unversioned build of this fork

If you have output from a pre-1.0.0 build of *this* project, two fixes alter the
numbers. Regenerate any baseline you compare against.

- **Segmentation no longer depends on the CPU core count.** Boxes seeded their
  `max_points_per_box` subsample with an id counted within each thread's share
  of the work, so the same box drew a different seed at every core count. On
  `data/test/example_clean.las`, 4 cores and 16 cores disagreed on 74,219 of
  673,517 point labels (11%) — enough to report 3 trees instead of 4, and to
  move DBH, height and volume with it. The seed is now keyed to the box's index
  in the full box array, so every core count agrees with a single-threaded run;
  the same comparison now differs on 2 labels, the `use_amp` fp16 jitter. Before
  this fix, one point cloud analysed on an 8-core laptop and a 32-core
  workstation gave different tree counts.
- **Multi-core subsampling no longer duplicates points.** With
  `num_cpu_cores > 1`, `subsample_point_cloud` asked a kd-tree for every point
  within one slice-width of each slice's lower bound. That search runs in both
  directions, so slices were twice as wide as the step between them and
  overlapped by half; every overlapped point was thinned twice and both copies
  kept. A 20,000 point cloud at `min_spacing=0.05` came back with 34,741 points,
  14,838 of them exact duplicates, at a minimum spacing of 0. Slices now tile the
  cloud exactly once, with a halo so points near a boundary are still thinned
  against their real neighbours. Only reachable with `subsample` enabled, which
  is off by default.

#### Fixed

- `wrapper_config.json` was missing its closing brace, so `batch_process.py`
  aborted with a `JSONDecodeError` before processing a single file. The config is
  repaired, and `load_config` now reports an unreadable config and continues on
  defaults instead of raising.
- `load_file` raised `UnboundLocalError: cannot access local variable
  'pointcloud'` for any extension that was not exactly `.las`, `.laz` or `.csv`
  in lower case — including `PLOT.LAS`. Extensions are matched case-insensitively
  and an unsupported format now raises an error naming the file and format.
- `batch_process.get_las_files` globbed case-sensitively and silently skipped
  uppercase `.LAS`/`.LAZ` files.
- The browser UI built conversion output paths with `str.replace`, which rewrites
  every occurrence in the path, not just the extension: `surveys/site.laz/plot.laz`
  became `surveys/site.las/plot.las`, pointing at a directory that does not exist.
- `training_monitor.py` read `except OSError or IndexError`, which evaluates to
  `except OSError` — `IndexError` was never caught. It also built its history
  path by concatenation onto a directory with no trailing separator, asking for
  `model/../modeltraining_history.csv`; the resulting `FileNotFoundError` was
  swallowed by that same handler, so the monitor redrew an empty figure forever.
- `subsample` crashed on inputs of fewer than two points
  (`Expected n_neighbors <= n_samples_fit`).

#### Changed

- The six plot builders in `visualization_utils.py` now log why a figure could
  not be built. They still return `None` so a page can skip the chart, but a
  missing column, an unreadable file and a plotly error are no longer
  indistinguishable from each other and from success.
- Corrected the `subsample` comment in `other_parameters.py`, which said
  "generally leave this on" beside a value of `0`.
- Removed 61 unused imports and a dead `DataLoader` compatibility shim. Verified
  no-ops: restoring them reproduced segmentation labels for all 673,517 points.
