# Changelog

Versioning is semantic: MAJOR for changes that alter measurements or break an
existing workflow, MINOR for new capability that leaves results alone, PATCH for
fixes with no effect on output. The version itself lives in `version.py`; run
`FSCT.bat version` to print it.

## 2.0.0

A MAJOR bump because two reported measurements change: `Volume_1` in
`tree_data.csv` was computed with a wrong formula, and `CCI_at_BH` was measured
against the wrong points. Everything else here leaves results untouched, and
was verified to: every CSV column and every LAS coordinate of
`data/test/example_clean.las` matches the 1.0.0 output after the two
measurement fixes, with the cylinder and stem files byte-identical. Regenerate
any baseline you compare against.

Benchmarks are on the same RTX 3050 / 16-core laptop as 1.0.0. It was far from
idle this time — a WSL VM, an IDE and browsers held 12+ GB between them — so
absolute times moved by 2x between runs and only back-to-back pairs are quoted.

### Large point clouds no longer crash

`data/test/ID0827_thin.las` — 5.1 million points, a 20 x 20 m plot — died at
the end of segmentation on a 16 GB machine:

    numpy._core._exceptions._ArrayMemoryError: Unable to allocate 623. MiB
    for an array with shape (5104837, 16) and data type float64

after the GPU had already done all its work. `choose_most_confident_label`
asked the kNN for distances it never used (a second `(N, 16)` array) and took
the median over one `(N, 16, 4)` gather, which numpy then needed a working copy
of — about 3.5 GB of temporaries for 5 million points. The medians are taken a
block of 250,000 points at a time now, with peak memory set by the block size
rather than the plot, and the per-box network output is released before the
labelled cloud is written. Labels are identical (checked on 673k points against
the old code). The same file now completes end to end: **156 s** on a quiet
machine, 270–280 s under load.

Two more allocation patterns that scale with the cloud were fixed on the way:
`load_file` rebuilt the whole cloud once per column it read (seven copies for
xyz + rgb + label), and `save_file` called `las.add_extra_dim` once per extra
column, each call copying the entire point record — 7 s of the post-processing
stage on the 5M-point file. Both now do one allocation.

### Measurement fixes

- **`Volume_1` was wrong twice over.** The frustum volume between consecutive
  cylinders read `(1/3) pi h (r1² + r1 + r2² + r2)` — adding a length to an
  area — where a frustum is `(1/3) pi h (r1² + r1·r2 + r2²)`; at r1 = 0.5 m,
  r2 = 0.3 m that overstates a segment by 2.3x. And the function's parameters
  were named `diameter_1`, `diameter_2` and halved on entry, while its only
  caller passes radii, so every radius was halved a second time. The two errors
  partly cancelled, which is why the numbers looked plausible. On the example
  plot tree 1's `Volume_1` moves from 1.61 m³ to 0.66 m³ — the old value
  exceeded a solid cylinder of the tree's own DBH over its full height, which no
  tapering stem can do; the new one sits beside `Volume_2` (0.53 m³), which is
  estimated independently from DBH and height and never went through this code.
- **`CCI_at_BH` used every tree's stem points.** The breast-height slice was
  cut from the whole plot's stem cloud, not the tree's, so a neighbouring stem
  falling in the 0.8–1.2 r annulus counted as coverage of this tree's
  circumference. It is now the tree's own points. On the sparse example plot
  the values happen not to change; on a dense plot they will.
- **Branch-to-parent interpolation matched the wrong parent.** Two bugs in the
  same block: `np.mean(parent_branch[:, :3])` with no axis returned a single
  number instead of a centroid, and `np.linalg.norm(...)` with no axis likewise,
  so `argmin` was always 0 and "the closest point of the current branch" was
  whatever row came first. Then the candidate angles were filtered in place and
  `argmin` of the *filtered* array was used to index the *unfiltered* candidate
  list, picking a different cylinder than the one with the smallest angle. Both
  fixed. They did not change the example plot's outputs, since they only bite
  when a branch has several parent candidates within `max_search_radius`.
- **The "VOLUME 2" annotation line was missing** from `text_point_cloud.las` in
  the default (uncropped) mode; only the tree-aware-cropping branch stacked it.
- **Division by zero on degenerate plots.** The coverage fractions and
  `Stems/ha` divided by the sample-cell count and the plot area with no guard;
  a plot whose DTM hull contained no grid cell, or with zero hull area, took the
  whole run down with `ZeroDivisionError` — once at the very end, after all the
  work was done. They now report 0.
- **`save_file` silently wrote nothing** for `.LAS` (upper case) or `.laz`: the
  extension check was `filename[-4:] == ".las"`, so neither branch matched and
  the function returned without an error. It now matches case-insensitively,
  like `load_file`, and raises on an unsupported extension.
- **Multi-core subsampling results depended on worker scheduling.** The
  subsampled slices were collected with `imap_unordered`, so the row order of
  the thinned cloud — and of everything derived from it — varied run to run.
  Now `imap`. Only reachable with `subsample` enabled.

One more found and deliberately left alone: the step that was meant to fold
DBSCAN's leftover "noise" skeleton points into their nearest cluster never did
anything. It wrote through a chained fancy index (into a temporary that was
discarded) *and* searched for neighbours only among the other unassigned points,
whose label is -1 by definition. Those points are silently dropped today.
`assign_unassigned_skeleton_points` in `other_parameters.py` makes the step work
as described and is **off by default**, because recovering skeleton points
changes which stems get cylinders fitted.

### Speed

Measurement stage on `example_clean.las`, 16 workers, back to back:
**20.0 s → 13.7 s**; the fastest of three later runs was 10.0 s against a
1.0.0 baseline of 18.2 s taken under the same conditions.

What the profile said, and what changed:

- **Worker start-up outweighed the work three to one.** Sampled across all 16
  workers, 146 s of CPU went on imports — each worker re-importing numpy,
  scipy, sklearn, pandas, networkx, laspy and hdbscan from a bare interpreter,
  about 9 s each under contention — against 54 s of actual clustering and
  circle fitting. pandas, networkx, laspy, skspatial and DBSCAN are only ever
  used by the parent and are now imported where they are used, taking a worker's
  import from 4.0 s to 3.3 s (sklearn, which hdbscan needs, is the floor).
  More usefully, the pool is now started **before segmentation** with a Pool
  initializer that forces those imports, so the workers come up while the GPU
  is busy and the CPU is idle, and are ready when measurement begins. This is
  gated on free memory: idle workers hold ~150 MB each and segmentation is
  where the parent's own memory peaks, so on a machine that is already paging
  the pool starts at the measurement stage as before. `prewarm_worker_pool` in
  `other_parameters.py` turns it off.
- **No more barriers between task batches.** Tasks were submitted in batches of
  128 with a wait for the whole batch after each, so one big stem cluster
  idled the other 15 workers until it finished — 32 s of worker CPU for
  cylinder fitting took 7.4 s of wall. Submission is now a sliding window
  (a semaphore keeps the same bound on tasks in flight), results are taken in
  completion order and slotted back by index, and tasks are submitted largest
  first. Every task is seeded per cluster, so neither order changes any output.
- **The DTM was re-triangulated twice per tree.** `griddata` builds a Delaunay
  triangulation on every call and measurement called it once per tree in the
  interpolation loop, once per tree when collecting tree data, and once per
  tree to find the base height — for one point. `DTMInterpolator` builds it
  once; identical values (`griddata(method="linear")` *is*
  `LinearNDInterpolator`). 20 per-tree calls against a 4,000-point DTM: 0.96 s
  → 0.22 s; against 20,000 points: 7.3 s → 0.75 s.
- **Skeleton-to-stem-point matching was brute force.** The cKDTrees for it were
  built with `leafsize=100000`, which makes the dual-tree query compare every
  skeleton point against every stem point. Leaf size changes only how the same
  neighbour sets are found: 3.65 s → 0.04 s on a synthetic 300k × 6k match,
  identical sets. The per-cluster boolean scans of the skeleton array and the
  per-cluster tree builds around it are replaced by one query and one grouping.
- **Quadratic array growth** in the tree-data loop (six arrays re-stacked per
  tree, two of them whole point clouds), the cylinder-interpolation loop (three
  nesting levels deep) and the skeleton visualisation, and the per-tree boolean
  scans of the vegetation and stem clouds — millions of rows scanned once per
  tree. All now group once and stack once.
- **Canopy / understorey / CWD cover** ran a Python loop over every 0.2 m grid
  cell with a generator-based convex-hull test and three `query_ball_point`
  calls per cell, each building an index list just to take its length. One
  matrix product for the hull test and three threaded counting queries.
- **DTM construction** — the per-cell radius-widening search over the terrain
  tree is done as four threaded counting queries over all cells at once, with
  the radii accumulated exactly as the loop did so boundary points land the
  same side; identical DTMs at 0.5 m and 0.3 m. The vegetation and stem
  sorting queries run on all cores.
- The point-based text annotations grew their array one point at a time in a
  nested Python loop over the character bitmap; `np.argwhere` gives the same
  points in the same order, 26x faster, seven times per tree.

### A dead worker no longer hangs the run

If a worker process is killed — the OS reclaiming memory is the usual cause —
`multiprocessing.Pool` starts a replacement but never re-runs the task the
dead one was holding, and `imap` waits for it forever. In the desktop app that
was a run that never finished, with no error anywhere. The result loop now
polls with a timeout and, if the pool's worker set has changed, raises an
error that says what happened and what to try. `close_pool` uses `terminate()`
rather than `close()` + `join()`, since after a lost task the pool's result
cache never drains and `join()` never returned either — the raised error was
never reaching the user. Verified by killing a worker mid-task: error in 30 s,
clean shutdown.

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
