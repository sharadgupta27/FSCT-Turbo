"""
FSCT Wrapper Tool
A user-friendly web interface for the Forest Structural Complexity Tool (FSCT)
"""

import streamlit as st
import collections
import functools
import os
import sys
import subprocess
import tempfile
import threading
import time
import shutil
import pandas as pd
import numpy as np
from datetime import datetime

# The FSCT modules in scripts/ import each other by bare name, so scripts/
# must be importable in its own right; the project root is added so this works
# from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.join(PROJECT_ROOT, 'scripts'), PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from version import __version__
import fsct_job

try:
    from setup_lastools import ensure_lastools, find_lastools_bin
    LASTOOLS_SETUP_AVAILABLE = True
except ImportError:
    LASTOOLS_SETUP_AVAILABLE = False


def _silence_proactor_disconnect_noise():
    """
    Stop Windows printing a scary traceback every time a browser tab closes.

    asyncio's Proactor transport calls sock.shutdown() when tearing a
    connection down and does not catch ConnectionResetError. If the peer has
    already gone - which is exactly what happens when you close or reload the
    Streamlit tab - Windows raises WinError 10054 and asyncio logs it as an
    unhandled exception in a callback:

        Exception in callback _ProactorBasePipeTransport._call_connection_lost
        ConnectionResetError: [WinError 10054] An existing connection was
        forcibly closed by the remote host

    Nothing is actually wrong: the socket is being discarded anyway and
    processing continues. This wraps the method to swallow that one error so
    the console stays readable.
    """
    if sys.platform != "win32":
        return
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
    except ImportError:
        return

    original = _ProactorBasePipeTransport._call_connection_lost
    if getattr(original, "_fsct_patched", False):
        return

    def _call_connection_lost(self, exc):
        try:
            original(self, exc)
        except (ConnectionResetError, ConnectionAbortedError):
            pass

    _call_connection_lost._fsct_patched = True
    _ProactorBasePipeTransport._call_connection_lost = _call_connection_lost


_silence_proactor_disconnect_noise()

# Import visualization utilities
try:
    from visualization_utils import create_3d_point_cloud_plot, create_height_distribution_plot, create_dbh_distribution_plot, create_tree_map_plot, create_dbh_height_scatter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class FSCTWrapper:
    """Main wrapper class for FSCT operations"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.lastools_path = None
        
    def __del__(self):
        """Cleanup temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def set_lastools_path(self, path):
        """Set the path to LAStools directory"""
        self.lastools_path = path
        
    def convert_laz_to_las(self, input_file, output_file=None):
        """Convert LAZ to LAS format using laszip"""
        if not output_file:
            # splitext, not replace - replace would also rewrite any earlier
            # ".laz" occurrence in the directory part of the path.
            output_file = os.path.splitext(input_file)[0] + '.las'
        
        if self.lastools_path:
            laszip_exe = os.path.join(self.lastools_path, 'laszip.exe')
            if os.path.exists(laszip_exe):
                cmd = [laszip_exe, '-i', input_file, '-o', output_file]
                subprocess.run(cmd, check=True)
                return output_file
        
        # Fallback to laspy
        try:
            import laspy
            las = laspy.read(input_file)
            las.write(output_file)
            return output_file
        except Exception as e:
            raise Exception(f"Failed to convert LAZ to LAS: {str(e)}")
    
    def convert_las_to_laz(self, input_file, output_file=None):
        """Convert LAS to LAZ format (compression)"""
        if not output_file:
            output_file = os.path.splitext(input_file)[0] + '.laz'
        
        if self.lastools_path:
            laszip_exe = os.path.join(self.lastools_path, 'laszip.exe')
            if os.path.exists(laszip_exe):
                cmd = [laszip_exe, '-i', input_file, '-o', output_file]
                subprocess.run(cmd, check=True)
                return output_file
        
        # Fallback to laspy
        try:
            import laspy
            las = laspy.read(input_file)
            las.write(output_file)
            return output_file
        except Exception as e:
            raise Exception(f"Failed to compress LAS to LAZ: {str(e)}")
    
    def resample_point_cloud(self, input_file, output_file, step_size=0.05):
        """Resample point cloud using LAStools or laspy"""
        if self.lastools_path:
            lasthin_exe = os.path.join(self.lastools_path, 'lasthin.exe')
            if os.path.exists(lasthin_exe):
                cmd = [lasthin_exe, '-i', input_file, '-o', output_file, 
                       '-step', str(step_size)]
                subprocess.run(cmd, check=True)
                return output_file
        
        # Fallback to custom resampling
        try:
            import laspy
            las = laspy.read(input_file)
            
            # Simple grid-based resampling
            xyz = np.vstack((las.x, las.y, las.z)).T
            
            # Create grid
            min_coords = xyz.min(axis=0)
            max_coords = xyz.max(axis=0)
            
            grid_size = step_size
            # int64: the flattened 3D cell id overflows the platform default
            # int32 on Windows for any sizeable cloud, which aliases distant
            # points onto one cell and silently discards real points.
            n_cells = np.floor((max_coords - min_coords) / grid_size).astype(np.int64) + 1

            # Assign points to grid cells
            cell_indices = np.floor((xyz - min_coords) / grid_size).astype(np.int64)
            cell_ids = cell_indices[:, 0] + cell_indices[:, 1] * n_cells[0] + cell_indices[:, 2] * n_cells[0] * n_cells[1]

            # Keep one point per cell
            unique_cells, unique_indices = np.unique(cell_ids, return_index=True)

            # Build a fresh header rather than reusing las.header, whose
            # point_count still describes the original cloud. Copying the whole
            # point record also preserves every attribute, instead of the two
            # that happened to be hardcoded here.
            new_header = laspy.LasHeader(
                point_format=las.header.point_format,
                version=str(las.header.version),
            )
            new_header.offsets = las.header.offsets
            new_header.scales = las.header.scales

            resampled_las = laspy.LasData(new_header)
            resampled_las.points = las.points[unique_indices]

            resampled_las.write(output_file)
            return output_file
            
        except Exception as e:
            raise Exception(f"Failed to resample point cloud: {str(e)}")
    
    def visualize_with_lastools(self, las_file):
        """Launch LAStools visualization"""
        if not self.lastools_path:
            raise Exception("LAStools path not set")
        
        lasview_exe = os.path.join(self.lastools_path, 'lasview.exe')
        if not os.path.exists(lasview_exe):
            raise Exception(f"lasview.exe not found at {lasview_exe}")
        
        subprocess.Popen([lasview_exe, las_file])
    
    def get_point_cloud_info(self, las_file):
        """Get basic information about a point cloud, from its header alone."""
        # laspy.read() loaded every point to report what the header already
        # says - minutes and gigabytes for a large cloud, on every page rerun.
        try:
            import laspy
            with laspy.open(las_file) as reader:
                header = reader.header
                mins, maxs = header.mins, header.maxs
                return {
                    'num_points': int(header.point_count),
                    'min_x': float(mins[0]),
                    'max_x': float(maxs[0]),
                    'min_y': float(mins[1]),
                    'max_y': float(maxs[1]),
                    'min_z': float(mins[2]),
                    'max_z': float(maxs[2]),
                    'point_format': header.point_format.id,
                    'version': f"{header.version.major}.{header.version.minor}"
                }
        except Exception as e:
            return {'error': str(e)}


def _read_file(path):
    with open(path, 'rb') as f:
        return f.read()


def deferred_download(path):
    """
    Download-button data for `path`, read only when the button is clicked.

    Passing the bytes read the whole file into memory on every page rerun,
    whether or not anyone downloaded it - for a point cloud, hundreds of MB
    per click anywhere on the page.
    """
    return functools.partial(_read_file, path)


class WebJob:
    """
    An FSCT run started from the browser, held in st.session_state.

    The run is an fsct_job child process with a reader thread collecting its
    output, so it outlives the script run that started it. Streamlit reruns
    the script on every widget interaction; a run tied to the script would
    either block the page until FSCT finished or be killed by the first
    slider touch. Only Stop ends this one.
    """

    def __init__(self, parameters):
        self.output_dir = fsct_job.output_dir_for(parameters['point_cloud_filename'])
        self._lock = threading.Lock()  # the reader appends while the page reads
        self.lines = collections.deque(maxlen=400)
        self.progress_line = ""
        self.stage = "Starting"
        self.failure = None
        self.stopped = False
        self.started = time.time()
        self.finished = None
        self.returncode = None
        self.process = fsct_job.start(parameters)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for text, transient in fsct_job.lines(self.process):
            if transient:
                self.progress_line = text
                continue
            self.progress_line = ""
            with self._lock:
                self.lines.append(text)
            if text.startswith(fsct_job.FAILURE_PREFIX):
                self.failure = text[len(fsct_job.FAILURE_PREFIX):]
            self.stage = fsct_job.stage_for(text) or self.stage
        self.returncode = self.process.returncode
        self.finished = time.time()

    def tail(self, count):
        with self._lock:
            return list(self.lines)[-count:]

    @property
    def running(self):
        return self.finished is None

    @property
    def succeeded(self):
        return self.returncode == 0 and not self.stopped

    def stop(self):
        self.stopped = True
        threading.Thread(target=fsct_job.stop, args=(self.process,), daemon=True).start()


# The stages a run passes through, in order, for the progress bar.
_STAGES = list(dict.fromkeys(stage for _, stage in fsct_job.STAGE_MARKERS))


def show_job(job):
    """Status, progress, recent output and a Stop button for a WebJob."""
    elapsed = (job.finished or time.time()) - job.started
    minutes, seconds = divmod(int(elapsed), 60)

    if job.running and job.stopped:
        st.info("Stopping the analysis...")
    elif job.running:
        done = _STAGES.index(job.stage) + 1 if job.stage in _STAGES else 0
        st.progress(done / (len(_STAGES) + 1), text=f"{job.stage}  ({minutes}:{seconds:02d})")
        if st.button("⏹ Stop analysis", key="stop_job"):
            job.stop()
            st.rerun()
    elif job.stopped:
        st.warning(f"Analysis stopped after {minutes}:{seconds:02d}. "
                   f"Its output folder is incomplete and can be deleted: {job.output_dir}")
    elif job.succeeded:
        st.success(f"✅ FSCT processing complete in {minutes}:{seconds:02d}. "
                   f"Output saved to: {job.output_dir}")
    else:
        st.error(f"❌ FSCT failed: {job.failure or f'the process exited with code {job.returncode}'}")

    tail = job.tail(14)
    if job.progress_line:
        tail.append(job.progress_line)
    if tail:
        st.code("\n".join(tail), language=None)


def main():
    """Main Streamlit application"""
    
    # Same icon the desktop app uses, so the browser tab matches. Falls back
    # to the emoji if icon.png has not been generated (see make_icon.py).
    icon_png = os.path.join(PROJECT_ROOT, "icon.png")
    st.set_page_config(
        page_title="FSCT Wrapper Tool",
        page_icon=icon_png if os.path.exists(icon_png) else "🌲",
        layout="wide"
    )
    
    st.title("🌲 Forest Structural Complexity Tool (FSCT) Wrapper")
    st.caption(f"Version {__version__}")
    st.markdown("---")
    
    # Initialize session state
    if 'wrapper' not in st.session_state:
        st.session_state.wrapper = FSCTWrapper()
    if 'lastools_path' not in st.session_state:
        # Pick up the copy that FSCT-Turbo.bat / setup_lastools.py unpacked,
        # so the user does not have to paste a path in by hand.
        detected = find_lastools_bin() if LASTOOLS_SETUP_AVAILABLE else None
        st.session_state.lastools_path = detected
        if detected:
            st.session_state.wrapper.set_lastools_path(detected)
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'processed_file_path' not in st.session_state:
        st.session_state.processed_file_path = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'job' not in st.session_state:
        st.session_state.job = None  # the current or last WebJob
    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = None  # which upload is already on disk

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LAStools path
        st.subheader("LAStools")
        if st.session_state.lastools_path:
            st.caption(f"✅ Found: `{st.session_state.lastools_path}`")
        else:
            st.caption("Optional - only the external 3D viewer needs it.")

        lastools_path = st.text_input(
            "Path to the LAStools 'bin' directory:",
            value=st.session_state.lastools_path or "",
            help="Normally filled in automatically by FSCT-Turbo.bat"
        )

        if st.button("Set LAStools Path"):
            if lastools_path and os.path.isdir(lastools_path):
                st.session_state.lastools_path = lastools_path
                st.session_state.wrapper.set_lastools_path(lastools_path)
                st.success("✅ LAStools path set successfully!")
            else:
                st.error("❌ Invalid path. Please check and try again.")

        if LASTOOLS_SETUP_AVAILABLE and not st.session_state.lastools_path:
            if st.button("⬇ Download LAStools"):
                with st.spinner("Downloading LAStools (about 60 MB)..."):
                    bin_dir = ensure_lastools(progress=None, quiet=True)
                if bin_dir:
                    st.session_state.lastools_path = bin_dir
                    st.session_state.wrapper.set_lastools_path(bin_dir)
                    st.success(f"✅ LAStools installed: {bin_dir}")
                else:
                    st.error("❌ Download failed. Install manually from https://rapidlasso.de/lastools/")


        st.markdown("---")
        
        # FSCT Parameters
        st.subheader("FSCT Parameters")
        
        with st.expander("Basic Settings", expanded=True):
            batch_size = st.slider("Batch Size", 1, 8, 2, help="Higher values need more VRAM")
            use_cpu_only = st.checkbox("Use CPU Only", value=False, help="Check if no GPU available")
            num_cpu_cores = st.number_input("CPU Cores", 0, os.cpu_count(), 0, 
                                           help="0 = use all cores")
        
        with st.expander("Plot Settings"):
            plot_radius = st.number_input("Plot Radius (m)", 0.0, 100.0, 0.0, 
                                         help="0 = no cropping")
            plot_radius_buffer = st.number_input("Plot Radius Buffer (m)", 0.0, 20.0, 0.0)
            tree_base_cutoff_height = st.number_input("Tree Base Cutoff Height (m)", 
                                                     0.0, 10.0, 5.0)
            ground_veg_cutoff_height = st.number_input("Ground Veg Cutoff Height (m)", 
                                                      0.0, 10.0, 3.0)
        
        with st.expander("Advanced Settings"):
            slice_thickness = st.slider("Slice Thickness", 0.05, 0.30, 0.15, 0.05)
            slice_increment = st.slider("Slice Increment", 0.01, 0.20, 0.05, 0.01)
            height_percentile = st.slider("Height Percentile", 90, 100, 100)
            sort_stems = st.checkbox("Sort Stems", value=True)
            generate_output_point_cloud = st.checkbox("Generate Output Point Cloud", value=True)
    
    # Main content area - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 File Upload & Processing", 
        "🔧 File Operations", 
        "🤖 Run Inference",
        "📊 Results & Visualization"
    ])
    
    # Tab 1: File Upload
    with tab1:
        st.header("Upload Point Cloud File")
        
        uploaded_file = st.file_uploader(
            "Choose a LAS or LAZ file",
            type=['las', 'laz'],
            help="Upload your forest point cloud file"
        )
        
        if uploaded_file is not None:
            # Save each upload once. This block runs on every rerun while the
            # uploader holds a file, and used to write a fresh timestamped
            # copy and reset processed_file_path each time - so a conversion
            # or resample was silently replaced by the raw upload on the very
            # next rerun, and every rerun added another full copy on disk.
            upload_key = (uploaded_file.file_id, uploaded_file.name, uploaded_file.size)
            if st.session_state.upload_key != upload_key:
                file_ext = os.path.splitext(uploaded_file.name)[1]
                temp_file_path = os.path.join(
                    st.session_state.wrapper.temp_dir,
                    f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
                )
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.upload_key = upload_key
                st.session_state.uploaded_file_path = temp_file_path
                st.session_state.processed_file_path = temp_file_path
            temp_file_path = st.session_state.uploaded_file_path

            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Show file info
            with st.spinner("Reading point cloud information..."):
                info = st.session_state.wrapper.get_point_cloud_info(temp_file_path)
                
                if 'error' not in info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Points", f"{info['num_points']:,}")
                    with col2:
                        st.metric("Point Format", info['point_format'])
                    with col3:
                        st.metric("LAS Version", info['version'])
                    
                    st.subheader("Bounding Box")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**X:** {info['min_x']:.2f} to {info['max_x']:.2f}")
                        st.write(f"**Y:** {info['min_y']:.2f} to {info['max_y']:.2f}")
                    with col2:
                        st.write(f"**Z:** {info['min_z']:.2f} to {info['max_z']:.2f}")
                        st.write(f"**Width:** {info['max_x'] - info['min_x']:.2f} m")
                        st.write(f"**Height:** {info['max_z'] - info['min_z']:.2f} m")
                else:
                    st.error(f"Error reading file: {info['error']}")
    
    # Tab 2: File Operations
    with tab2:
        st.header("File Operations")
        
        if st.session_state.processed_file_path is None:
            st.warning("⚠️ Please upload a file first in the 'File Upload & Processing' tab.")
        else:
            current_file = st.session_state.processed_file_path
            st.info(f"Current file: {os.path.basename(current_file)}")
            
            col1, col2, col3 = st.columns(3)
            
            # Convert
            with col1:
                st.subheader("Convert Format")
                if current_file.lower().endswith('.las'):
                    if st.button("Convert to LAZ (Compress)", use_container_width=True):
                        with st.spinner("Converting to LAZ..."):
                            try:
                                # splitext, not replace: replace rewrites every
                                # ".las" in the path, including in directory names.
                                output_file = os.path.splitext(current_file)[0] + '.laz'
                                st.session_state.wrapper.convert_las_to_laz(current_file, output_file)
                                st.session_state.processed_file_path = output_file
                                st.success("✅ Converted to LAZ!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                elif current_file.lower().endswith('.laz'):
                    if st.button("Convert to LAS (Decompress)", use_container_width=True):
                        with st.spinner("Converting to LAS..."):
                            try:
                                # splitext, not replace - see the note above.
                                output_file = os.path.splitext(current_file)[0] + '.las'
                                st.session_state.wrapper.convert_laz_to_las(current_file, output_file)
                                st.session_state.processed_file_path = output_file
                                st.success("✅ Converted to LAS!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
            
            # Resample
            with col2:
                st.subheader("Resample")
                step_size = st.number_input("Step Size (m)", 0.01, 1.0, 0.05, 0.01,
                                           help="Grid cell size for resampling")
                
                if st.button("Resample Point Cloud", use_container_width=True):
                    with st.spinner("Resampling..."):
                        try:
                            # splitext, not replace: replace rewrote every
                            # ".las" in the path, directory names included.
                            stem, file_ext = os.path.splitext(current_file)
                            output_file = f"{stem}_resampled{file_ext}"
                            st.session_state.wrapper.resample_point_cloud(
                                current_file, output_file, step_size
                            )
                            st.session_state.processed_file_path = output_file
                            st.success("✅ Resampled!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            # Download
            with col3:
                st.subheader("Download")
                if os.path.exists(current_file):
                    st.download_button(
                        label="Download Current File",
                        data=deferred_download(current_file),
                        file_name=os.path.basename(current_file),
                        mime="application/octet-stream",
                        use_container_width=True
                    )
    
    # Tab 3: Run Inference
    with tab3:
        st.header("Run FSCT Inference")
        
        if st.session_state.processed_file_path is None:
            st.warning("⚠️ Please upload a file first.")
        else:
            current_file = st.session_state.processed_file_path
            job = st.session_state.job
            running = job is not None and job.running

            # FSCT reads LAS only. A LAZ file is decompressed when the run
            # starts - not on every rerun, which used to undo "Convert to LAZ"
            # the moment that button's rerun reached this tab.
            run_file = os.path.splitext(current_file)[0] + '.las' \
                if current_file.lower().endswith('.laz') else current_file
            st.info(f"📁 Processing file: {os.path.basename(run_file)}"
                    + (" (decompressed from LAZ when the run starts)" if run_file != current_file else ""))

            parameters = fsct_job.job_parameters(
                point_cloud_filename=run_file,
                plot_radius=plot_radius,
                plot_radius_buffer=plot_radius_buffer,
                batch_size=batch_size,
                num_cpu_cores=num_cpu_cores if num_cpu_cores > 0 else os.cpu_count(),
                use_CPU_only=use_cpu_only,
                slice_thickness=slice_thickness,
                slice_increment=slice_increment,
                sort_stems=1 if sort_stems else 0,
                height_percentile=height_percentile,
                tree_base_cutoff_height=tree_base_cutoff_height,
                generate_output_point_cloud=1 if generate_output_point_cloud else 0,
                ground_veg_cutoff_height=ground_veg_cutoff_height,
            )

            with st.expander("View All Parameters"):
                st.json(parameters)

            if st.button("🚀 Run FSCT Inference", type="primary", use_container_width=True,
                         disabled=running):
                try:
                    if run_file != current_file:
                        with st.spinner("Converting LAZ to LAS..."):
                            st.session_state.wrapper.convert_laz_to_las(current_file, run_file)
                        st.session_state.processed_file_path = run_file
                    st.session_state.job = WebJob(parameters)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Could not start FSCT: {str(e)}")

            if job is not None:
                # Redrawn once a second while the run is going. When it ends,
                # rerun the whole page once so the Results tab picks it up.
                @st.fragment(run_every=1.0 if running else None)
                def job_panel():
                    current = st.session_state.job
                    show_job(current)
                    if not current.running and st.session_state.get('job_shown_done') is not current:
                        st.session_state.job_shown_done = current
                        if current.succeeded:
                            st.session_state.output_dir = current.output_dir
                        st.rerun()

                job_panel()
    
    # Tab 4: Results & Visualization
    with tab4:
        st.header("Results & Visualization")
        
        if st.session_state.output_dir is None:
            st.warning("⚠️ No results available. Please run inference first.")
        else:
            output_dir = st.session_state.output_dir
            
            if os.path.exists(output_dir):
                st.success(f"📂 Output directory: {output_dir}")
                
                # List output files
                st.subheader("Output Files")
                output_files = []
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        output_files.append(os.path.join(root, file))
                
                if output_files:
                    # Display CSV results
                    csv_files = [f for f in output_files if f.endswith('.csv')]
                    
                    if csv_files:
                        st.subheader("📊 Measurement Results")
                        
                        for csv_file in csv_files:
                            csv_name = os.path.basename(csv_file)
                            with st.expander(f"📄 {csv_name}"):
                                try:
                                    df = pd.read_csv(csv_file)
                                    st.dataframe(df, use_container_width=True)
                                    
                                    # Download button
                                    csv_data = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label=f"Download {csv_name}",
                                        data=csv_data,
                                        file_name=csv_name,
                                        mime='text/csv'
                                    )
                                except Exception as e:
                                    st.error(f"Error reading {csv_name}: {str(e)}")
                    
                    # Display images
                    image_files = [f for f in output_files if f.endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if image_files:
                        st.subheader("🖼️ Generated Figures")
                        
                        cols = st.columns(2)
                        for idx, img_file in enumerate(image_files):
                            with cols[idx % 2]:
                                st.image(img_file, caption=os.path.basename(img_file), 
                                       use_container_width=True)
                    
                    # LAS/LAZ files for visualization
                    las_files = [f for f in output_files if f.endswith(('.las', '.laz'))]
                    
                    if las_files:
                        st.subheader("🌲 Point Cloud Visualization")
                        
                        selected_las = st.selectbox(
                            "Select point cloud to visualize:",
                            las_files,
                            format_func=lambda x: os.path.basename(x)
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📊 View with LAStools", use_container_width=True):
                                if st.session_state.lastools_path:
                                    try:
                                        st.session_state.wrapper.visualize_with_lastools(selected_las)
                                        st.success("✅ LAStools viewer launched!")
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                                else:
                                    st.error("❌ Please set LAStools path in the sidebar first.")
                        
                        with col2:
                            st.download_button(
                                label="Download Point Cloud",
                                data=deferred_download(selected_las),
                                file_name=os.path.basename(selected_las),
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                    
                    # Summary statistics
                    plot_summary_file = os.path.join(output_dir, 'plot_summary.csv')
                    tree_data_file = os.path.join(output_dir, 'tree_data.csv')
                    
                    if os.path.exists(plot_summary_file):
                        st.subheader("📈 Plot Summary Statistics")
                        
                        try:
                            plot_summary = pd.read_csv(plot_summary_file)

                            # The columns FSCT actually writes, as on the
                            # desktop Results page. This used to look for
                            # 'Number of Trees', 'Mean Tree Height' and 'Total
                            # Basal Area', which FSCT never emits, so only Mean
                            # DBH ever showed - in metres, labelled cm.
                            metrics = [
                                ('Num Trees in Plot', 'Trees detected', '{:,.0f}'),
                                ('Stems/ha', 'Stems per hectare', '{:,.0f}'),
                                ('Mean DBH', 'Mean DBH (m)', '{:.3f}'),
                                ('Mean Height', 'Mean height (m)', '{:.1f}'),
                                ('Total Volume 1', 'Stem volume (m³)', '{:.2f}'),
                            ]
                            for column, (key, label, fmt) in zip(st.columns(len(metrics)), metrics):
                                if key in plot_summary.columns and not pd.isna(plot_summary[key].values[0]):
                                    with column:
                                        st.metric(label, fmt.format(plot_summary[key].values[0]))
                        
                        except Exception as e:
                            st.error(f"Error reading plot summary: {str(e)}")
                    
                    # Enhanced visualizations
                    if VISUALIZATION_AVAILABLE and os.path.exists(tree_data_file):
                        st.subheader("📊 Interactive Visualizations")
                        
                        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                            "Tree Map", "Height Distribution", "DBH Distribution", "DBH vs Height"
                        ])
                        
                        with viz_tab1:
                            with st.spinner("Creating tree map..."):
                                tree_map = create_tree_map_plot(tree_data_file)
                                if tree_map:
                                    st.plotly_chart(tree_map, use_container_width=True)
                                else:
                                    st.warning("Could not create tree map")
                        
                        with viz_tab2:
                            with st.spinner("Creating height distribution..."):
                                height_dist = create_height_distribution_plot(tree_data_file)
                                if height_dist:
                                    st.plotly_chart(height_dist, use_container_width=True)
                                else:
                                    st.warning("Could not create height distribution")
                        
                        with viz_tab3:
                            with st.spinner("Creating DBH distribution..."):
                                dbh_dist = create_dbh_distribution_plot(tree_data_file)
                                if dbh_dist:
                                    st.plotly_chart(dbh_dist, use_container_width=True)
                                else:
                                    st.warning("Could not create DBH distribution")
                        
                        with viz_tab4:
                            with st.spinner("Creating DBH vs Height scatter..."):
                                dbh_height = create_dbh_height_scatter(tree_data_file)
                                if dbh_height:
                                    st.plotly_chart(dbh_height, use_container_width=True)
                                else:
                                    st.warning("Could not create scatter plot")
                    
                    # 3D Point Cloud Visualization
                    if VISUALIZATION_AVAILABLE and las_files:
                        st.subheader("🌲 3D Point Cloud Preview")
                        st.info("Note: Only a sample of points is shown for performance. Use LAStools for full visualization.")
                        
                        selected_pc = st.selectbox(
                            "Select point cloud for 3D preview:",
                            las_files,
                            format_func=lambda x: os.path.basename(x),
                            key='3d_preview'
                        )
                        
                        max_points = st.slider("Max points to display", 10000, 200000, 50000, 10000)
                        color_option = st.selectbox("Color by:", ['z', 'intensity', 'classification'])
                        
                        if st.button("Generate 3D Preview", use_container_width=True):
                            with st.spinner("Generating 3D visualization..."):
                                fig_3d = create_3d_point_cloud_plot(selected_pc, max_points, color_option)
                                if fig_3d:
                                    st.plotly_chart(fig_3d, use_container_width=True)
                                else:
                                    st.error("Failed to create 3D visualization")
                
                else:
                    st.warning("No output files found.")
            else:
                st.error(f"Output directory not found: {output_dir}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center'>
            <p>FSCT Wrapper Tool v{__version__} | Forest Structural Complexity Tool</p>
            <p>Original FSCT: <a href='https://github.com/SKrisanski/FSCT' target='_blank'>
            https://github.com/SKrisanski/FSCT</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
