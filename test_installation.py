"""
FSCT installation test.

Run this to verify the environment is complete. FSCT-Turbo.bat runs it
automatically as the final step of setup, and on demand via
"FSCT-Turbo.bat verify".

    python test_installation.py
"""

import importlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from version import __version__


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_result(test_name, passed, message=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]  {test_name}")
    if message:
        print(f"          {message}")


def test_python_version():
    """FSCT targets Python 3.11. 3.9 is past end of life; 3.13 has no
    torch-cluster wheels for the pinned torch version."""
    version = sys.version_info
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    if version[:2] < (3, 9):
        return False, message + " (too old - 3.11 recommended)"
    if version[:2] != (3, 11):
        return True, message + " (3.11 is the tested version)"
    return True, message


def test_import(module_name, package_name=None):
    """Import a module and report its version."""
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        # Not just ImportError: a torch/CUDA DLL mismatch raises OSError, and
        # that is exactly the failure mode this script exists to catch.
        return False, f"{type(error).__name__}: {error}"
    version = getattr(module, "__version__", "unknown")
    return True, f"{package_name or module_name} {version}"


def test_files(label, required_files):
    missing = [f for f in required_files if not os.path.exists(os.path.join(PROJECT_ROOT, f))]
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, f"All {label} present"


def test_linalg():
    """
    Exercise the LAPACK routines FSCT actually calls.

    Importing numpy is not enough. A mismatched BLAS can import cleanly and
    then kill the process the first time a LAPACK routine runs - conda-forge's
    MKL 2026.1.0 did exactly that here, taking down np.linalg.svd and lstsq
    with 0xC06D007F and no Python traceback. That breaks cylinder fitting
    (fit_cylinder uses svd, skimage's CircleModel uses lstsq) while every
    import check still passes, so check the calls themselves.
    """
    import subprocess

    # Run out-of-process: a bad BLAS aborts the interpreter rather than
    # raising, so it would take this script down with it.
    code = (
        "import numpy as np;"
        "A=np.random.rand(40,3);b=np.random.rand(40);"
        "np.linalg.lstsq(A,b,rcond=None);"
        "np.linalg.svd(np.random.rand(30,30));"
        "np.linalg.inv(np.random.rand(20,20));"
        "print('ok')"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    except Exception as error:
        return False, f"Could not run the check: {error}"

    if r.returncode != 0 or b"ok" not in r.stdout:
        return False, (
            f"numpy LAPACK call crashed (exit {r.returncode}). The BLAS build is broken - "
            "reinstall with the OpenBLAS variant: "
            'conda install -n lidar -c conda-forge "libblas=*=*openblas"'
        )

    import numpy as np
    return True, f"svd / lstsq / inv work (numpy {np.__version__})"


def test_torch_geometric_ops():
    """
    scripts/model.py needs fps, radius and knn_interpolate. These live in
    torch-cluster, which is a separate wheel from torch-geometric and is the
    piece most likely to be missing after a partial install.
    """
    try:
        import torch
        from torch_geometric.nn import fps, radius, knn_interpolate  # noqa: F401

        pos = torch.rand(64, 3)
        batch = torch.zeros(64, dtype=torch.long)
        idx = fps(pos, batch, ratio=0.5)
        radius(pos, pos[idx], 0.3, batch, batch[idx], max_num_neighbors=8)
        return True, "fps / radius / knn operators work"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def test_cuda():
    try:
        import torch
    except Exception as error:
        return False, f"Could not import torch: {error}"

    if torch.cuda.is_available():
        return True, f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}"

    # Distinguish "no GPU" from "GPU present but a CPU-only torch got
    # installed", which is the exact state the old installer produced.
    if torch.version.cuda is None:
        import shutil
        import subprocess

        has_gpu = False
        if shutil.which("nvidia-smi"):
            try:
                has_gpu = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=15
                ).returncode == 0
            except Exception:
                has_gpu = False
        if has_gpu:
            return False, (
                "An Nvidia GPU is present but a CPU-only build of PyTorch is installed. "
                "Re-run FSCT-Turbo.bat setup /force to rebuild the environment."
            )
        return True, "CPU-only build, no Nvidia GPU detected (fine, just slower)"

    return True, "CUDA build installed but no GPU visible (will run on CPU)"


def test_lastools():
    try:
        from setup_lastools import find_lastools_bin
    except ImportError:
        return True, "setup_lastools.py not found (optional)"

    bin_dir = find_lastools_bin()
    if bin_dir:
        return True, bin_dir
    return True, "Not installed (optional - run: python setup_lastools.py)"


def main():
    print_header(f"FSCT Installation Test - v{__version__}")

    required_passed = True

    print("\n[1] Python")
    passed, message = test_python_version()
    print_result("Version", passed, message)
    required_passed &= passed

    print("\n[2] Deep learning stack")
    # torchvision is intentionally absent - see the note in FSCT-Turbo.bat.
    for module, name in [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("torch_cluster", "torch-cluster"),
    ]:
        passed, message = test_import(module, name)
        print_result(name, passed, message)
        required_passed &= passed

    passed, message = test_torch_geometric_ops()
    print_result("Geometric operators", passed, message)
    required_passed &= passed

    print("\n[3] Scientific stack")
    passed, message = test_linalg()
    print_result("numpy LAPACK (BLAS health)", passed, message)
    required_passed &= passed

    for module, name in [
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("skimage", "scikit-image"),
        ("matplotlib", "matplotlib"),
        ("networkx", "networkx"),
        ("hdbscan", "hdbscan"),
        ("skspatial", "scikit-spatial"),
        ("community", "python-louvain"),
    ]:
        passed, message = test_import(module, name)
        print_result(name, passed, message)
        required_passed &= passed

    print("\n[4] Point clouds and reporting")
    for module, name in [
        ("laspy", "laspy"),
        ("lazrs", "lazrs"),
        ("mdutils", "mdutils"),
        ("markdown", "Markdown"),
    ]:
        passed, message = test_import(module, name)
        print_result(name, passed, message)
        required_passed &= passed

    print("\n[5] FSCT files")
    passed, message = test_files("core files", [
        "scripts/run_tools.py",
        "scripts/inference.py",
        "scripts/preprocessing.py",
        "scripts/model.py",
        "scripts/measure.py",
        "model/model.pth",
        "version.py",
        "fsct_job.py",  # both interfaces run the pipeline through it
    ])
    print_result("Core files", passed, message)
    required_passed &= passed

    print("\n[6] FSCT imports")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    sys.path.insert(0, PROJECT_ROOT)
    passed, message = test_import("scripts.run_tools", "scripts.run_tools")
    print_result("FSCT pipeline", passed, message if passed else message)
    required_passed &= passed

    print("\n[7] Optional")
    for label, check in [
        ("GPU / CUDA", test_cuda),
        ("LAStools", test_lastools),
        ("CustomTkinter", lambda: test_import("customtkinter", "CustomTkinter")),
    ]:
        passed, message = check()
        print_result(label, passed, message)
        # CUDA misconfiguration is worth failing on; the rest are genuinely
        # optional and must not fail the run.
        if label == "GPU / CUDA":
            required_passed &= passed

    print_header("Summary")
    if required_passed:
        print("\n  All required checks passed.\n")
        print("  Start FSCT with:  FSCT-Turbo.bat gui")
    else:
        print("\n  Some required checks FAILED.\n")
        print("  Run 'FSCT-Turbo.bat install /force' to rebuild the environment.")
    print("\n" + "=" * 60 + "\n")

    return 0 if required_passed else 1


if __name__ == "__main__":
    sys.exit(main())
