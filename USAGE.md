# Installing and running FSCT-Turbo

This is the complete guide to installing FSCT-Turbo and running it. For what
FSCT-Turbo adds over the original FSCT, see [README.md](README.md). For what FSCT
measures, what its output files contain and what each inherited parameter means,
see the [original FSCT README](https://github.com/SKrisanski/FSCT#readme).
For what changed between releases, see [CHANGELOG.md](CHANGELOG.md).

Check which version you have with `FSCT-Turbo.bat version`.

---

## TL;DR

1. Install [Miniforge](https://github.com/conda-forge/miniforge/releases/latest).
2. Double-click **`FSCT-Turbo.bat`**.

That's it. The first run installs everything by itself: conda environment,
PyTorch, the desktop app and LAStools, and asks nothing. It takes 15-25 minutes
and needs about 6 GB, then starts the desktop app. Every run after that starts
the desktop app straight away.

---

## 1. Installation

### FSCT-Turbo.bat

`FSCT-Turbo.bat` is the only batch file. Normally you just double-click it, but it also
takes commands:

| Command | What it does |
|---|---|
| `FSCT-Turbo.bat` | Set up if needed, then launch the desktop app |
| `FSCT-Turbo.bat gui` | Launch the desktop app, skipping the setup check |
| `FSCT-Turbo.bat setup` | Re-run setup, keeping the environment |
| `FSCT-Turbo.bat setup /force` | Rebuild the environment from scratch |
| `FSCT-Turbo.bat verify` | Run the installation checks |
| `FSCT-Turbo.bat lastools` | Re-download LAStools |
| `FSCT-Turbo.bat help` | Command list |

Setup completion is recorded in a `.fsct_setup` stamp file next to the script.
Delete it to force setup to run again on the next launch. The stamp also carries
a version number, so bumping the package set in `FSCT-Turbo.bat` makes existing
installs refresh themselves automatically.

The desktop app, the command line and batch processing all use the **same**
`lidar` conda environment.

### Version matrix

These versions are pinned deliberately. The `torch` / `torch-geometric` /
`torch-cluster` triple has to agree, or the segmentation model will not load.

| Package | Version | Notes |
|---|---|---|
| Python | 3.11 | 3.9 is end-of-life; 3.13 has no `torch-cluster` wheel for torch 2.5.1 |
| PyTorch | 2.5.1 | `cu121` build when an Nvidia GPU is present, `cpu` otherwise |
| torch-geometric | 2.6.1 | |
| torch-cluster | 1.6.3 | prebuilt wheel from `data.pyg.org`, no compiler needed |
| numpy | >= 1.26 | 2.x is fine |
| BLAS | OpenBLAS | forced with `libblas=*=*openblas`; see below |

`torch-scatter` and `torch-sparse` are **not** installed. torch-geometric 2.x
uses native torch scatter ops; only `torch-cluster` is still required, for the
`fps` / `radius` / `knn` operators in `scripts/model.py`.

**`torchvision` is not installed either.** Nothing in FSCT imports it and
torch-geometric does not need it, but the image DLLs bundled in its pip wheel
shadow the ones conda-forge's Pillow links against. The result is
`ImportError: DLL load failed while importing _imaging` for everything that
touches PIL, including matplotlib and therefore the whole report writer.

**OpenBLAS is forced over MKL.** conda-forge otherwise pulls Intel MKL, and MKL
2026.1.0 fails to resolve a delay-loaded export on this platform:
`numpy.linalg.svd` and `lstsq` kill the process outright with `0xC06D007F` and no
Python traceback. That takes out cylinder fitting completely.

**Why PyTorch comes from pip, not conda.** Mixing the `pytorch` channel with
conda-forge lets the solver satisfy the constraints with a **CPU-only** `pytorch`
build alongside a **CUDA** `torchaudio` build. It "succeeds", then fails at
runtime. Installing the torch stack from the official PyTorch wheel index avoids
the whole class of problem.

### Manual installation

If you would rather not use `FSCT-Turbo.bat`:

```bat
conda create -n lidar python=3.11 pip -y -c conda-forge

conda install -n lidar -y -c conda-forge --override-channels ^
    "libblas=*=*openblas" ^
    "numpy>=1.26" pandas scipy scikit-learn scikit-image matplotlib ^
    networkx laspy lazrs-python tqdm joblib hdbscan jinja2 rdflib pillow ^
    pywavelets fsspec requests

conda run -n lidar --no-capture-output python -m pip install ^
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

conda run -n lidar --no-capture-output python -m pip install torch-geometric==2.6.1
conda run -n lidar --no-capture-output python -m pip install torch-cluster ^
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

conda run -n lidar --no-capture-output python -m pip install ^
    mdutils markdown python-louvain scikit-spatial customtkinter

conda run -n lidar --no-capture-output python setup_lastools.py
conda run -n lidar --no-capture-output python test_installation.py
```

Replace `cu121` with `cpu` in both URLs if you have no Nvidia GPU.

`requirements.txt` is the single requirements file: core pipeline and desktop
app. `FSCT-Turbo.bat` runs `pip install -r requirements.txt` as its final
dependency step, so the same file that describes a plain-venv install is the one
the supported installer exercises.

Every version bound in it is deliberately open at the top. conda-forge and the
PyTorch wheel index install newer builds than PyPI would resolve to, and a
capped bound makes that final pip pass pull PyPI wheels over them, which is how
an earlier `numpy<2.2, pandas<3.0` pin would have downgraded a working
environment. If you tighten a bound, check it with:

```bat
conda run -n lidar python -m pip install -r requirements.txt --dry-run
```

On a correctly built environment that must report nothing to install.

### LAStools

`FSCT-Turbo.bat setup` runs `setup_lastools.py`, which downloads the official
distribution (about 60 MB), unpacks it to `third_party/LAStools/` and writes the
path to its `bin` directory into `gui_config.json`. Nothing needs to be set on
the Settings page.

To (re)run it by hand:

```bat
FSCT-Turbo.bat lastools                                      :: re-download
conda run -n lidar python setup_lastools.py            :: install only if missing
```

LAStools is **optional**. It powers only the external 3D viewer button; reading,
writing, converting and resampling all go through `laspy`. If the download fails,
FSCT still works.

Licensing: LAStools is a mixed distribution. `laszip` and `lasview`, the tools
FSCT uses, are free. Some other executables in the same package are commercial
and watermark their output without a licence key. See
<https://rapidlasso.de/lastools/>.

### Verifying

```bat
FSCT-Turbo.bat verify
```

This is the same check `FSCT-Turbo.bat setup` runs at the end. It distinguishes "no GPU
present" from "GPU present but a CPU-only PyTorch got installed", and it
exercises the `torch-cluster` operators rather than just importing the package.

To check the code rather than the installation, run the regression tests
(about a minute). Setting `FSCT_E2E=1` adds a full run of the example plot,
which checks its segmentation to the exact point count:

```bat
conda run -n lidar python -m unittest discover -s tests

set FSCT_E2E=1
conda run -n lidar python -m unittest discover -s tests
```

### System requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 | Windows 10/11 |
| RAM | 8 GB | 16 GB+ |
| CPU | 4 cores | 8+ cores |
| GPU | none (CPU mode) | Nvidia, 6 GB+ VRAM |
| Disk | 10 GB free | 20 GB+ free |

Linux works; macOS is untested.

---

## 2. Running the desktop app

```bat
FSCT-Turbo.bat gui
```

A sidebar on the left selects one of five pages. The file you are working on is
named in the header, so it stays visible whichever page you are on; that
header chip is read-only. Choosing a file happens in one place, the Point Cloud
page; `Ctrl+O` jumps there and opens the picker.

The **Appearance** control at the bottom of the sidebar switches between Light,
Dark and System, and the choice is remembered in `gui_config.json`.

### Point Cloud

Load and inspect a point cloud. Click **Browse...** and pick a LAS or LAZ file.
Point count, density, extent, height range and file size appear as tiles, with
the full header underneath.

Only the header is read, so even a multi-gigabyte file opens instantly. If the
header has no bounding box recorded (plenty of exporters leave it zeroed), the
real extent is measured from the points in the background, and the panel says
so.

### Tools

- **LAZ → LAS (decompress)** and **LAS → LAZ (compress)** for format conversion.
- **Resample** to reduce point density, with a grid step from 0.01 m to 1.0 m.
  0.05 m is a good general value.
- **Open in lasview** launches the external 3D viewer, and **Open containing
  folder** browses the results.
- **Save a clean copy** rewrites the file through laspy onto a fresh header.
  Use it when lasview refuses a file with

      ERROR: point has size of 28 but items only add up to 36 bytes
             please upgrade to the latest release of LAStools

  That message is misleading: upgrading LAStools does not help. The file
  carries a stale `laszip encoded` record left behind by an earlier format
  conversion, describing a point layout the file no longer uses. LAStools
  trusts it and gives up; laspy ignores it, which is why FSCT processes such a
  file without complaint. The clean copy keeps **every** point (it is a
  repair, not a resample) and preserves the coordinates, intensity, GPS time,
  classification, scales, offsets, version, point format and the
  georeferencing VLR.
- The operation log shows timestamped status for everything done here.

### Analysis

Parameters on the left, a live processing console on the right. Each parameter
has a slider and a typed value, so you can drag roughly or enter an exact
number, and a one-line note explaining what it changes.

**Compute**: batch size, CPU core count, force-CPU mode.
**Plot**: plot radius, plot buffer, tree base height, ground veg height.
**Measurement**: slice thickness, slice increment, height percentile, sort
stems, write the segmented output cloud.

All of these are explained in the [original FSCT
README](https://github.com/SKrisanski/FSCT#user-parameters); the ones FSCT-Turbo
adds are in [README.md](README.md#new-parameters). Start with
the defaults; the suggested batch size is chosen from your GPU's memory.

Press **Run FSCT analysis** and watch the console, which reports the current
stage and elapsed time as it goes. **Save log...** writes the console to a text
file. The app switches to Results when the run finishes.

**Stop** ends the run, worker processes included, after asking you to confirm.
The output folder is left incomplete; delete it or run again. Closing the
window during a run stops it the same way.

### Results

Headline tiles (trees detected, stems per hectare, mean DBH, mean height, total
stem volume) read from `plot_summary.csv`, a list of every generated file with
type, size and date, and buttons to refresh, open the output folder, or show
the full plot summary as a table.

Double-click a file in the list (or select it and press **Open selected**) to
open it. Point clouds go to lasview, since Windows has no default association
for `.las`/`.laz`; everything else (the report figures, CSVs and the HTML or
markdown report) opens in whatever application you have associated with it.

### Settings

Set the path to LAStools' `bin` directory if the automatic download did not run,
and save it for future sessions. Also shows which Python, FSCT modules and
PyTorch build the app is actually running against, which is the first thing to
check when something fails to import.

---

## 3. Running from the command line

### A single plot, or a hand-picked set

Edit the parameters at the top of `scripts/run.py`, then:

```bat
conda run -n lidar --no-capture-output python scripts/run.py
```

It opens a file picker so you can select one or more `.las` files. To skip the
picker, replace the `file_mode()` call with `directory_mode()` (which finds every
`.las` under a directory, ignoring anything in a folder with `FSCT_output` in the
name), or with a plain list of paths.

Each of the five stages can be switched off independently in the `FSCT(...)`
call (`preprocess`, `segmentation`, `postprocessing`, `measure_plot`,
`make_report`), but each needs the previous one to have been run already. This is
useful when iterating: run preprocessing and segmentation once, then re-run just
the measurement stage.

### A whole directory, unattended

```bat
conda run -n lidar python batch_process.py <input_directory>
conda run -n lidar python batch_process.py <input_directory> --no-recursive
conda run -n lidar python batch_process.py <input_directory> --combine-only
conda run -n lidar python batch_process.py <input_directory> --config my_params.json
```

Results from every plot are combined into a single timestamped CSV. Parameters
come from `wrapper_config.json` beside `batch_process.py`, or the file given
with `--config`; any parameter a config leaves out takes its default. The exit
code is 0 when every file succeeded, 1 when any failed and 2 for a misspelt
parameter in the config, so a script or scheduled task can check it.

### Combining results from earlier runs

```bat
conda run -n lidar python scripts/combine_multiple_output_CSVs.py
```

Gets all `plot_summary.csv` files and combines them into one CSV, written to the
highest common directory of the selected point clouds.

---

## 4. Choosing parameters

### By file size

| Points | What to do |
|---|---|
| < 50M | Use as-is |
| 50-100M | Consider resampling to 0.05 m |
| > 100M | Resample to 0.05-0.10 m first |

### By forest type

| | Dense forest | Sparse forest |
|---|---|---|
| Slice thickness | 0.15 m | 0.20 m |
| Slice increment | 0.05 m | 0.05 m |

Noisy data above the canopy: set height percentile to 98.

### By hardware

- **With a GPU**: keep the defaults. The app picks a batch size from the
  detected VRAM; raising it beyond that is usually slower, not faster.
- **CPU only**: tick "Use CPU Only" and drop the batch size to 1. Expect
  inference to take considerably longer. The results are the same as on a GPU
  apart from floating-point rounding: on the example plot 84 of 673,517 labels
  differ, and the four trees get identical DBH and height. The batch size never
  changes the results, only the speed.
- **Limited RAM**: reduce the CPU core count and resample the cloud first.

### Where the parameters live

- The two UIs expose the common ones directly.
- `scripts/run.py` holds them for command-line use.
- `scripts/other_parameters.py` holds the advanced ones: segmentation box
  geometry, clustering thresholds, the random seed, and the two knobs that
  control cylinder fitting cost (`circle_fit_trials` and
  `circle_fit_max_points`).

---

## 5. Example data

One test file ships with the project: `data/test/example.las`, the example
plot from the original FSCT. A complete run of `data/test/example.las` takes well under a minute on a laptop
RTX 3050.

---

## 6. Troubleshooting

**"Failed to create environment" / solver conflicts**
Run `FSCT-Turbo.bat setup /force`. A part-built environment from an earlier failed run
cannot reliably be repaired in place.

**`ModuleNotFoundError` for `torch_geometric`, `mdutils`, `community` or `markdown`**
A previous install failed part-way and reported success anyway. Run
`FSCT-Turbo.bat setup /force`.

**`torch.cuda.is_available()` is False on a machine with an Nvidia GPU**
A CPU-only build got installed. Recreate the environment with
`FSCT-Turbo.bat setup /force`.

**The process dies during cylinder fitting with `0xC06D007F` and no traceback**
An MKL build of numpy got in. Recreate the environment; setup forces OpenBLAS.

**CUDA out of memory**
Reduce the batch size to 1, or tick "Force CPU only", or resample the point
cloud first. Closing other GPU applications helps.

**`ImportError: DLL load failed while importing _imaging`**
`torchvision` got installed and its bundled image DLLs are shadowing Pillow's.
Uninstall it, or recreate the environment.

**`[ASN1: NOT_ENOUGH_DATA]` when downloading LAStools**
A malformed certificate in the Windows certificate store breaks OpenSSL 3.x for
every Python HTTPS request. `setup_lastools.py` works around it using certifi's
CA bundle. If it still fails, download LAStools manually and point the Settings
page at its `bin` folder.

**"lasview.exe not found"**
LAStools isn't installed or isn't configured. Run `FSCT-Turbo.bat lastools`, or install
it from <https://rapidlasso.de/lastools/> and set the path on the Settings page.

**`ValueError: substring not found` from `get_fsct_path`**
Fixed. That function used to slice the working directory at the first "FSCT"
substring, so it only worked if the project folder was named exactly `FSCT`. It
now resolves paths from the location of `scripts/tools.py`.

**`TypeError: percentile() got an unexpected keyword argument 'interpolation'`**
Fixed. `interpolation=` was renamed to `method=` in numpy 1.22 and removed in
2.0, which killed the measurement stage just before tree heights were computed.

**Permission error writing `plot_summary.csv`**
You have it open in Excel. Close it.

**Processing seems stuck at "Starting multithreaded cylinder fitting"**
It should not be any more; see the measurement-stage figures in
[README.md](README.md#performance). If it still is, the plot is probably very
large; lower `circle_fit_trials` in `scripts/other_parameters.py`.

**No trees found**
A plot with no trees completes normally: `tree_data.csv` is empty, the plot
summary reports 0 trees and the report says "No stems found". If you expected
trees, check `segmented.las` in a viewer; if the stems came out labelled as
vegetation, the cloud is likely too low resolution for the segmentation model.

---

## 7. Getting help

- Upstream project and issues: https://github.com/SKrisanski/FSCT
- Example files in `data/`
- `FSCT-Turbo.bat verify` for a quick health check of the installation
