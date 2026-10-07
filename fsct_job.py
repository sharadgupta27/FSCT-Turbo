"""
One FSCT run, in its own process, so it can be stopped.

FSCT() is a single long call with no cancellation points, and its measurement
stage fans out to a pool of worker processes. Run inside a UI's process, that
meant Stop could only explain that nothing could be stopped, and closing the
window had to os._exit() past a half-finished pool. Run here instead, as a
child process, a job can be killed - workers included - and its output,
workers' output included, arrives through one pipe.

Both interfaces use this module, and batch_process.py uses its parameters:

    parameters = job_parameters(batch_size=4)   # defaults + overrides + advanced
    job = start(parameters)
    for line, transient in lines(job):          # transient: a "\\r" progress line
        ...
    stop(job)                                   # from any thread; kills the workers too

A job still running when the UI's process exits is stopped with it.

Run directly, it is the child side: `python fsct_job.py parameters.json`.
"""

import atexit
import codecs
import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback
import weakref

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# The user parameters every run needs, with the defaults of upstream FSCT's
# scripts/run.py. This is the one copy: the desktop app and batch_process.py
# both start from it, so a parameter added here reaches every
# entry point. wrapper_config.json overrides it for batch runs only.
DEFAULT_PARAMETERS = dict(
    plot_centre=None,  # [X, Y], or None for the centre of the cloud's bounding box
    plot_radius=0,  # 0 = do not crop
    plot_radius_buffer=0,  # non-zero with plot_radius: tree-aware plot cropping
    batch_size=2,
    num_cpu_cores=0,  # 0 = all cores
    use_CPU_only=False,
    slice_thickness=0.15,
    slice_increment=0.05,
    sort_stems=1,
    height_percentile=100,
    tree_base_cutoff_height=5,
    generate_output_point_cloud=1,
    ground_veg_cutoff_height=3,
    veg_sorting_range=1.5,
    stem_sorting_range=1,  # unused, as in the original: stems sort within veg_sorting_range
    taper_measurement_height_min=0,
    taper_measurement_height_max=30,
    taper_measurement_height_increment=0.2,
    taper_slice_thickness=0.4,
    delete_working_directory=True,
    minimise_output_size_mode=0,
)

# Substrings that identify which stage of the pipeline is running, matched in
# order against each line of output; first hit wins.
STAGE_MARKERS = (
    ("Pre-processing point cloud", "Preprocessing"),
    ("Performing inference", "Semantic segmentation"),
    ("Semantic segmentation done", "Post-processing"),
    ("Loading segmented point cloud", "Post-processing"),
    ("Making DTM", "Building DTM"),
    ("Post processing done", "Measuring plot"),
    ("Making and clustering slices", "Measuring: slices"),
    ("cylinder fitting", "Measuring: cylinder fitting"),
    ("Sorting Cylinders", "Measuring: sorting stems"),
    ("Cylinder interpolation", "Measuring: interpolation"),
    ("Sorting vegetation", "Measuring: vegetation"),
    ("Measuring plot done", "Writing report"),
)

# The child's last line when the pipeline raises, so a UI can show the error
# without parsing a traceback.
FAILURE_PREFIX = "FSCT job failed: "


def job_parameters(**overrides):
    """DEFAULT_PARAMETERS, then `overrides`, then scripts/other_parameters.py."""
    if os.path.join(PROJECT_ROOT, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    from other_parameters import other_parameters

    unknown = set(overrides) - set(DEFAULT_PARAMETERS) - {"point_cloud_filename"}
    if unknown:
        raise KeyError(f"Unknown FSCT parameter(s): {', '.join(sorted(unknown))}")
    parameters = dict(DEFAULT_PARAMETERS)
    parameters.update(overrides)
    parameters.update(other_parameters)
    return parameters


def stage_for(line):
    """The pipeline stage a line of output announces, or None."""
    lowered = line.lower()
    for marker, stage in STAGE_MARKERS:
        if marker.lower() in lowered:
            return stage
    return None


def output_dir_for(point_cloud_filename):
    """Where FSCT writes a cloud's results."""
    return os.path.splitext(point_cloud_filename)[0] + "_FSCT_output"


_running = weakref.WeakSet()


def start(parameters):
    """Start FSCT on `parameters` in a child process and return its Popen."""
    fd, path = tempfile.mkstemp(prefix="fsct_job_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(parameters, f)

    kwargs = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True  # so stop() can signal the whole group
    # On Windows the child shares the console, so Ctrl+C there ends the job
    # along with the UI rather than leaving it running on its own.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    job = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(__file__), path],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        **kwargs,
    )
    _running.add(job)
    return job


def lines(job):
    """
    Yield (line, transient) for each line of the job's output until it exits.

    FSCT draws progress counters with print("\\r", i, "/", n, end=""), tens of
    thousands of them in a run. Text ended by a lone "\\r" is yielded with
    transient True, so a console can overwrite the previous one instead of
    appending. "\\r\\n" is an ordinary line end: on Windows the child's
    text-mode stdout writes every newline that way.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    while True:
        data = job.stdout.read1(65536)
        buffer += decoder.decode(data, final=not data)
        while True:
            newline, carriage = buffer.find("\n"), buffer.find("\r")
            if newline == -1 and carriage == -1:
                break
            if carriage == -1 or (newline != -1 and newline < carriage):
                line, buffer = buffer[:newline], buffer[newline + 1:]
                yield line.rstrip(), False
            elif carriage + 1 == len(buffer) and data:
                break  # the "\n" of a "\r\n" may be in the next read
            elif buffer.startswith("\n", carriage + 1):
                line, buffer = buffer[:carriage], buffer[carriage + 2:]
                yield line.rstrip(), False
            else:
                line, buffer = buffer[:carriage], buffer[carriage + 1:]
                if line.strip():
                    yield line.rstrip(), True
        if not data:
            break
    if buffer.strip():
        yield buffer.rstrip(), False
    job.stdout.close()
    job.wait()


def stop(job):
    """Kill the job and every worker process it started. Safe to call twice."""
    if job is None or job.poll() is not None:
        return
    if os.name == "nt":
        # /T takes the measurement workers down with it; job.kill() alone
        # would leave them running, holding memory, with nothing to report to.
        subprocess.run(
            ["taskkill", "/PID", str(job.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(job.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        job.wait(timeout=15)
    except subprocess.TimeoutExpired:
        job.kill()


@atexit.register
def _stop_all():
    for job in list(_running):
        stop(job)


def _main(parameters_file):
    with open(parameters_file, encoding="utf-8") as f:
        parameters = json.load(f)
    os.remove(parameters_file)

    for path in (os.path.join(PROJECT_ROOT, "scripts"), PROJECT_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from run_tools import FSCT

        FSCT(
            parameters=parameters,
            preprocess=True,
            segmentation=True,
            postprocessing=True,
            measure_plot=True,
            make_report=True,
            clean_up_files=False,
        )
    except Exception as error:
        traceback.print_exc()
        print(f"\n{FAILURE_PREFIX}{type(error).__name__}: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1]))
