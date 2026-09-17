from preprocessing import Preprocessing
from inference import SemanticSegmentation
from post_segmentation_script import PostProcessing
from measure import MeasureTree
from report_writer import ReportWriter
import glob
import tkinter as tk
import tkinter.filedialog as fd
import os
import sys


def _available_memory_gb():
    """
    Physical memory currently available to new allocations, in GB, or None
    if it cannot be determined.

    psutil comes in with torch, but is not a declared dependency of this
    project, so there is a Windows fallback via the Win32 API and a Linux one
    via sysconf.
    """
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / 1e9
        except Exception:
            pass
    else:
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except (AttributeError, ValueError, OSError):
            pass
    return None


def FSCT(
    parameters,
    preprocess=True,
    segmentation=True,
    postprocessing=True,
    measure_plot=True,
    make_report=False,
    clean_up_files=False,
):
    print("Current point cloud being processed: ", parameters["point_cloud_filename"])
    if parameters["num_cpu_cores"] == 0:
        print("Using default number of CPU cores (all of them).")
        parameters["num_cpu_cores"] = os.cpu_count()
    print("Processing using ", parameters["num_cpu_cores"], "/", os.cpu_count(), " CPU cores.")

    if preprocess:
        preprocessing = Preprocessing(parameters)
        preprocessing.preprocess_point_cloud()
        del preprocessing

    try:
        if measure_plot and segmentation and parameters.get("prewarm_worker_pool", True):
            # Start the measurement stage's worker pool now, so the workers do
            # their start-up while the GPU is busy with segmentation.
            #
            # A spawned worker on Windows begins as a bare interpreter and has
            # to import numpy, scipy, sklearn and hdbscan before it can take
            # its first task. Profiled with 16 workers that came to about 9 s
            # each - three times the CPU the workers then spent on the actual
            # clustering and circle fitting of a small plot. Segmentation runs
            # on the GPU with the CPU mostly idle, and takes longer than the
            # workers need to come up, so overlapping the two hides that
            # start-up completely. MeasureTree picks the same pool up by its
            # process count and shuts it down when it is done; the finally
            # below covers a run that fails before it gets there.
            #
            # Only when there is room for it. Idle workers hold about 150 MB
            # each once their imports are in, and starting them here puts that
            # on top of segmentation, which is where this process's own memory
            # peaks (the model, the CUDA context and the per-box network
            # output all at once). On a machine that was already paging, that
            # was the difference between finishing and dying with
            # "Unable to allocate 43.7 MiB" in the middle of inference. When
            # memory is short the pool is simply started later, at the
            # beginning of measurement, exactly as it was before.
            workers = parameters["num_cpu_cores"]
            needed_gb = workers * 0.2 + 3.0
            available_gb = _available_memory_gb()
            if available_gb is None:
                print("Could not read free memory; the measurement workers will start at the measurement stage.")
            elif available_gb < needed_gb:
                print(
                    f"{available_gb:.1f} GB free, below the {needed_gb:.1f} GB wanted to start {workers} "
                    "measurement workers during segmentation; they will start at the measurement stage instead."
                )
            else:
                MeasureTree._get_pool(workers)

        if segmentation:
            sem_seg = SemanticSegmentation(parameters)
            sem_seg.inference()
            del sem_seg

        if postprocessing:
            object_1 = PostProcessing(parameters)
            object_1.process_point_cloud()
            del object_1

        if measure_plot:
            measure1 = MeasureTree(parameters)
            measure1.run_measurement_extraction()
            del measure1
    finally:
        MeasureTree.close_pool()

    if make_report:
        report_writer = ReportWriter(parameters)
        report_writer.make_report()
        del report_writer

    if clean_up_files:
        report_writer = ReportWriter(parameters)
        report_writer.clean_up_files()
        del report_writer


def directory_mode():
    root = tk.Tk()
    point_clouds_to_process = []
    directory = fd.askdirectory(parent=root, title="Choose directory")
    unfiltered_point_clouds_to_process = glob.glob(directory + "/**/*.las", recursive=True)
    for i in unfiltered_point_clouds_to_process:
        if "FSCT_output" not in i:
            point_clouds_to_process.append(i)
    root.destroy()
    return point_clouds_to_process


def file_mode():
    root = tk.Tk()
    point_clouds_to_process = fd.askopenfilenames(
        parent=root, title="Choose files", filetypes=[("LAS", "*.las"), ("LAZ", "*.laz"), ("CSV", "*.csv")]
    )
    root.destroy()
    return point_clouds_to_process
