"""
FSCT Desktop Application

A CustomTkinter interface for the Forest Structural Complexity Tool.

Layout: a fixed sidebar selects one of five pages, which are stacked in the
same grid cell and raised on demand. That replaces the old ttk.Notebook -
tabs put the navigation, the page title and the content controls all on one
crowded strip, and there was nowhere to show which file is loaded.

Threading: every long operation runs on a daemon worker. Tk is not thread
safe, so workers never touch a widget. They push text onto plain deques and a
single repeating timer on the main loop drains them (see _pump_output). The
few one-shot callbacks that do need the main thread go through
_on_main_thread.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import collections
import math
import struct
import threading
import os
import sys
import subprocess
import time
import json
from datetime import datetime
import webbrowser

try:
    import customtkinter as ctk
except ImportError:  # pragma: no cover - environment problem, not a code path
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror(
        "CustomTkinter is missing",
        "The FSCT desktop app needs the 'customtkinter' package.\n\n"
        "Run  FSCT.bat setup  to repair the environment, or install it with:\n"
        "    pip install customtkinter",
    )
    sys.exit(1)

# The FSCT modules in scripts/ import each other by bare name ("from tools
# import ..."), so scripts/ has to be importable in its own right. The project
# root is added too, so this works no matter which directory the GUI is
# launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.join(PROJECT_ROOT, 'scripts'), PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from version import __version__

_fsct_lock = threading.Lock()
_fsct_modules = None
FSCT_IMPORT_ERROR = None


def load_fsct_modules():
    """
    Import the FSCT pipeline, once, on demand.

    Deliberately not done at module import. scripts.run_tools pulls in torch,
    torch_geometric, scikit-learn and scipy, which measured 55 s on a cold
    cache - and because the import ran before Tk was even touched, the app
    showed absolutely nothing for that whole time. Now the window paints
    immediately and a background warm-up does this while the user is choosing
    a file.

    Returns (FSCT, other_parameters); raises ImportError if the environment is
    incomplete.
    """
    global _fsct_modules, FSCT_IMPORT_ERROR
    with _fsct_lock:
        if _fsct_modules is not None:
            return _fsct_modules
        try:
            from scripts.run_tools import FSCT
            from scripts.other_parameters import other_parameters
        except ImportError as error:
            FSCT_IMPORT_ERROR = error
            raise
        _fsct_modules = (FSCT, other_parameters)
        return _fsct_modules


try:
    from setup_lastools import ensure_lastools, find_lastools_bin
    LASTOOLS_SETUP_AVAILABLE = True
except ImportError:
    LASTOOLS_SETUP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Appearance
#
# Every colour is a (light, dark) pair, which is what CustomTkinter expects.
# Anything drawn outside CustomTkinter - the ttk.Treeview, the Text widgets -
# has to be recoloured by hand whenever the mode changes, see _restyle_native.
# ---------------------------------------------------------------------------
APP_BG = ("#EEF2EF", "#141719")
SIDEBAR_BG = ("#FFFFFF", "#1A1E21")
CARD_BG = ("#FFFFFF", "#1F2429")
CARD_BORDER = ("#E1E8E3", "#2B3238")
INSET_BG = ("#F4F7F5", "#191D20")

TEXT = ("#152119", "#E7EDEA")
TEXT_MUTED = ("#5E6B64", "#98A39E")

ACCENT = ("#1E7A4F", "#2FA36B")
ACCENT_HOVER = ("#186340", "#3CBB7D")
ACCENT_SOFT = ("#E4F2EA", "#22322A")

DANGER = ("#B3261E", "#E5806E")
CONSOLE_BG = ("#12161A", "#0E1114")
CONSOLE_FG = ("#D7E0DA", "#D7E0DA")

FONT_TITLE = ("Segoe UI", 21, "bold")
FONT_H1 = ("Segoe UI", 17, "bold")
FONT_H2 = ("Segoe UI", 13, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 11)
FONT_TINY = ("Segoe UI", 10)
FONT_STAT = ("Segoe UI", 18, "bold")
FONT_MONO = ("Consolas", 11)

# Substrings that identify which stage of the pipeline is running, so the
# Analysis page can show progress instead of an anonymous spinner. Matched in
# order against each console line, first hit wins.
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


def shade(pair):
    """Resolve a (light, dark) pair against the current appearance mode."""
    try:
        dark = ctk.get_appearance_mode() == "Dark"
    except Exception:
        dark = False
    return pair[1] if dark else pair[0]


# ---------------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------------
class Card(ctk.CTkFrame):
    """Rounded panel with an optional title and one-line description."""

    def __init__(self, master, title=None, description=None, **kwargs):
        kwargs.setdefault("corner_radius", 12)
        kwargs.setdefault("fg_color", CARD_BG)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", CARD_BORDER)
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        row = 0

        if title:
            ctk.CTkLabel(
                self, text=title, font=FONT_H2, text_color=TEXT, anchor="w"
            ).grid(row=row, column=0, sticky="ew", padx=18, pady=(16, 0))
            row += 1

        self._description = None
        if description:
            self._description = ctk.CTkLabel(
                self, text=description, font=FONT_SMALL, text_color=TEXT_MUTED,
                anchor="w", justify="left", wraplength=400,
            )
            self._description.grid(row=row, column=0, sticky="ew", padx=18, pady=(2, 0))
            # A fixed wraplength gets clipped in the narrow three-across layout
            # on the Tools page, so follow the card's real width instead.
            # add="+" matters: CustomTkinter binds <Configure> on every widget
            # to redraw its rounded background, and a plain bind() would
            # replace that binding rather than run alongside it.
            self.bind("<Configure>", self._rewrap_description, add="+")
            row += 1

        # columnspan 2: column 1 is left free so a caller can put an action
        # button level with the title (see the Header details card).
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=row, column=0, columnspan=2, sticky="nsew",
                       padx=18, pady=(12, 16))
        self.body.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(row, weight=1)

    def _rewrap_description(self, event):
        # event.width is in Tk pixels, but CustomTkinter multiplies whatever
        # wraplength it is given by the widget scaling factor. Divide it out,
        # or the text wraps 25% too late on a 125% display and spills past the
        # card edge. 36 is the label's left and right padding.
        try:
            scaling = ctk.ScalingTracker.get_widget_scaling(self) or 1.0
        except Exception:
            scaling = 1.0
        self._description.configure(
            wraplength=max(120, int((event.width - 36) / scaling))
        )


class StatTile(ctk.CTkFrame):
    """One headline number with a caption underneath."""

    def __init__(self, master, caption, value="--"):
        super().__init__(master, corner_radius=10, fg_color=INSET_BG)
        self.grid_columnconfigure(0, weight=1)

        self.value_label = ctk.CTkLabel(
            self, text=value, font=FONT_STAT, text_color=ACCENT, anchor="w"
        )
        self.value_label.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 0))

        ctk.CTkLabel(
            self, text=caption, font=FONT_TINY, text_color=TEXT_MUTED, anchor="w"
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))

    def set(self, value):
        self.value_label.configure(text=value)


class ParamRow:
    """
    Label, slider and a typed value for one parameter.

    The slider snaps to `step` and the entry accepts a typed value, so coarse
    dragging and exact input both work. Bad or out-of-range input reverts to
    the last good value on commit rather than raising.
    """

    def __init__(self, parent, row, label, variable, from_, to, step=1,
                 unit="", hint=None):
        self.variable = variable
        self.from_ = from_
        self.to = to
        self.step = step
        self.is_int = isinstance(variable, tk.IntVar)
        self.decimals = 0 if self.is_int or step >= 1 else max(
            0, -int(round(math.log10(step)))
        )

        parent.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            parent, text=label, font=FONT_BODY, text_color=TEXT, anchor="w"
        ).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=(8, 0))

        self.entry_var = tk.StringVar(value=self._format(variable.get()))
        entry = ctk.CTkEntry(
            parent, textvariable=self.entry_var, width=78, height=28,
            font=FONT_BODY, justify="right",
        )
        entry.grid(row=row, column=2, sticky="e", pady=(8, 0))
        entry.bind("<Return>", self._commit_entry)
        entry.bind("<FocusOut>", self._commit_entry)

        if unit:
            ctk.CTkLabel(
                parent, text=unit, font=FONT_TINY, text_color=TEXT_MUTED,
                width=26, anchor="w",
            ).grid(row=row, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        steps = max(1, int(round((to - from_) / step)))
        self.slider = ctk.CTkSlider(
            parent, from_=from_, to=to, number_of_steps=steps,
            variable=variable, command=self._on_slide, height=18,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            progress_color=ACCENT,
        )
        self.slider.grid(row=row + 1, column=0, columnspan=4, sticky="ew", pady=(2, 0))

        self.hint_label = None
        if hint:
            self.hint_label = ctk.CTkLabel(
                parent, text=hint, font=FONT_TINY, text_color=TEXT_MUTED,
                anchor="w", justify="left", wraplength=330,
            )
            self.hint_label.grid(row=row + 2, column=0, columnspan=4,
                                 sticky="ew", pady=(1, 6))

    @staticmethod
    def rows_used(hint):
        return 3 if hint else 2

    def _format(self, value):
        if self.is_int:
            return str(int(round(float(value))))
        return f"{float(value):.{self.decimals}f}"

    def _snap(self, value):
        snapped = round(float(value) / self.step) * self.step
        snapped = min(max(snapped, self.from_), self.to)
        if self.is_int:
            return int(round(snapped))
        # Round to the step's own precision, otherwise the value shows
        # artefacts like 0.15000000000000002.
        return round(snapped, self.decimals)

    def _on_slide(self, value):
        # Setting `variable` moves the slider, but CustomTkinter suppresses the
        # command for variable-driven updates, so this cannot recurse.
        snapped = self._snap(value)
        self.variable.set(snapped)
        self.entry_var.set(self._format(snapped))

    def set_value(self, value):
        """Set the value from code. The entry only tracks the slider through
        the callbacks above, so it has to be told as well."""
        snapped = self._snap(value)
        self.variable.set(snapped)
        self.entry_var.set(self._format(snapped))

    def set_hint(self, text):
        if self.hint_label is not None:
            self.hint_label.configure(text=text)

    def _commit_entry(self, _event=None):
        try:
            snapped = self._snap(float(self.entry_var.get()))
        except (TypeError, ValueError):
            snapped = self.variable.get()
        self.variable.set(snapped)
        self.entry_var.set(self._format(snapped))


def accent_button(master, text, command, **kwargs):
    kwargs.setdefault("height", 36)
    kwargs.setdefault("corner_radius", 8)
    kwargs.setdefault("font", FONT_H2)
    return ctk.CTkButton(
        master, text=text, command=command,
        fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#FFFFFF",
        **kwargs,
    )


def ghost_button(master, text, command, **kwargs):
    kwargs.setdefault("height", 36)
    kwargs.setdefault("corner_radius", 8)
    kwargs.setdefault("font", FONT_BODY)
    return ctk.CTkButton(
        master, text=text, command=command,
        fg_color="transparent", hover_color=ACCENT_SOFT,
        text_color=TEXT, border_width=1, border_color=CARD_BORDER,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# LAS sanity check
# ---------------------------------------------------------------------------
def laszip_vlr_mismatch(path):
    """
    Detect a stale LASzip VLR left behind by a format conversion.

    A LASzip VLR (user id "laszip encoded", record id 22204) lists the items a
    point is built from. If a tool rewrites the points in a different format
    but leaves that VLR behind, the item sizes no longer match the header's
    point_data_record_length. laspy ignores the VLR on an uncompressed file so
    it reads happily, but LAStools trusts it and refuses to open the file with

        ERROR: point has size of 28 but items only add up to 36 bytes
               please upgrade to the latest release of LAStools (with LASzip)

    which sends people off upgrading LAStools when the file is the problem.

    Returns (declared_size, header_size) when they disagree, otherwise None.
    Never raises: this only decorates an error path.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(8192)

        if len(head) < 227 or head[:4] != b"LASF":
            return None

        header_size = struct.unpack_from("<H", head, 94)[0]
        num_vlrs = struct.unpack_from("<I", head, 100)[0]
        record_length = struct.unpack_from("<H", head, 105)[0]

        pos = header_size
        for _ in range(min(num_vlrs, 64)):
            if pos + 54 > len(head):
                return None
            record_id = struct.unpack_from("<H", head, pos + 18)[0]
            body_length = struct.unpack_from("<H", head, pos + 20)[0]
            body = head[pos + 54:pos + 54 + body_length]

            if record_id == 22204 and len(body) >= 34:
                num_items = struct.unpack_from("<H", body, 32)[0]
                declared = 0
                offset = 34
                for _item in range(num_items):
                    if offset + 6 > len(body):
                        return None
                    declared += struct.unpack_from("<HHH", body, offset)[1]
                    offset += 6
                if declared != record_length:
                    return declared, record_length
                return None

            pos += 54 + body_length
    except (OSError, struct.error, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# stdout/stderr capture
# ---------------------------------------------------------------------------
class ConsoleStream:
    """
    Splits a text stream into lines for the in-app processing console.

    FSCT reports progress with print("\\r", i, "/", n, end=""), so naive
    line-per-write logging produced tens of thousands of near-identical lines
    and made the window crawl. Text after a carriage return is flagged
    transient, and the console replaces the previous transient line instead of
    appending to it.

    `original`, if given, also receives everything - but the app passes None.
    Mirroring to the launching terminal duplicates the whole run in two places
    and the GUI console is the one being read; use "Save log..." to keep a
    copy. Note this only redirects the parent process: scripts/measure.py and
    scripts/tools.py fan work out to spawned worker processes that hold their
    own console handle, so anything *they* print still lands in the terminal
    and cannot be intercepted from here.
    """

    def __init__(self, sink, original=None):
        self._sink = sink
        self._original = original
        self._buffer = ""

    def write(self, text):
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass

        self._buffer += text
        while True:
            newline = self._buffer.find("\n")
            carriage = self._buffer.find("\r")
            if newline == -1 and carriage == -1:
                break
            if newline != -1 and (carriage == -1 or newline < carriage):
                line, self._buffer = self._buffer[:newline], self._buffer[newline + 1:]
                self._sink(line.rstrip(), False)
            else:
                line, self._buffer = self._buffer[:carriage], self._buffer[carriage + 1:]
                if line.strip():
                    self._sink(line.rstrip(), True)

    def flush(self):
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class FSCTStandaloneApp:
    """Main application class."""

    PAGES = (
        ("point_cloud", "Point Cloud", "Load a LAS or LAZ file and inspect it"),
        ("tools", "Tools", "Convert, resample and view point clouds"),
        ("analysis", "Analysis", "Configure and run the FSCT pipeline"),
        ("results", "Results", "Outputs from the last completed run"),
        ("settings", "Settings", "LAStools, appearance and about"),
    )

    def __init__(self, root):
        self.root = root
        self.root.title("FSCT - Forest Structural Complexity Tool")

        self.current_file = tk.StringVar()
        self.lastools_path = tk.StringVar()
        self.output_directory = tk.StringVar()
        self.appearance_mode = tk.StringVar(value="System")
        self.processing = False
        self.threads = []
        self._start_time = None

        # Workers append here; _pump_output drains them on the main loop.
        self._console_queue = collections.deque()
        self._log_queue = collections.deque()
        self._callback_queue = collections.deque()
        self._console_transient = False
        self._pending_stage = None
        self._file_header = None
        # Set by the warm-up worker once scripts.run_tools has been imported.
        self.fsct_ready = False
        self.fsct_error = None

        self.pages = {}
        self.nav_buttons = {}
        self.stat_tiles = {}
        self.result_tiles = {}
        self._logo_image = None

        self.load_config()
        try:
            ctk.set_appearance_mode(self.appearance_mode.get())
        except Exception:
            ctk.set_appearance_mode("System")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.fit_to_screen()
        self.set_window_icon()
        self.build_ui()
        self.show_page("point_cloud")

        # Ctrl+O is the other way into the file picker, so send it through the
        # Point Cloud page too rather than letting a file be swapped in while
        # some other page is on screen.
        self.root.bind("<Control-o>", lambda _e: self._browse_from_point_cloud())
        self.root.after(120, self._pump_output)
        self.root.after(1000, self._tick_clock)
        self._start_worker(self._warm_up)

    # -- deferred imports ---------------------------------------------------
    def _warm_up(self):
        """
        Import FSCT and probe the GPU off the startup path.

        Everything here is slow (torch alone is tens of seconds on a cold
        cache) and none of it is needed to draw the window, so it happens in
        the background while the user is picking a file.
        """
        try:
            load_fsct_modules()
            error = None
        except Exception as import_error:
            error = import_error

        batch, gpu_note = self.recommended_batch_size()
        self._on_main_thread(self._apply_warm_up, error, batch, gpu_note,
                             self.torch_description())

    def _apply_warm_up(self, error, batch, gpu_note, torch_note):
        self.fsct_ready = error is None
        self.fsct_error = error

        self.batch_row.set_value(batch)
        self.batch_row.set_hint(gpu_note)

        status = "loaded" if self.fsct_ready else f"NOT AVAILABLE - {error}"
        self.env_label.configure(
            text=f"Python {sys.version.split()[0]}\n"
                 f"FSCT modules: {status}\n"
                 f"{torch_note}"
        )
        if not self.fsct_ready:
            self.update_status("FSCT modules failed to import - see Settings")

    # -- window chrome ------------------------------------------------------
    def set_window_icon(self):
        ico = os.path.join(PROJECT_ROOT, 'icon.ico')
        if not os.path.exists(ico):
            return
        try:
            # Twice on purpose. tkinter's wm_iconbitmap ignores `bitmap`
            # whenever `default` is passed, so one call dresses this window and
            # the other supplies the icon for Toplevels opened later, such as
            # the plot summary table.
            self.root.iconbitmap(ico)
            self.root.iconbitmap(default=ico)
        except tk.TclError:
            pass

    def fit_to_screen(self):
        """
        Size the window so it fits the display, and centre it.

        CustomTkinter multiplies the *size* in a geometry() call by the display
        scaling factor - 1.25 on a 125% display - but leaves the position
        alone. Two traps follow from that, and the previous version hit both:

        * "1280x840" really asks for 1600x1050, which does not fit a 1536x864
          laptop screen, so the status bar and the Run button ended up under
          the taskbar.
        * winfo_width() reports the already-scaled size, so feeding it back
          into geometry() scales it a second time. Centring that way grew the
          window to 1924x1055 - wider and taller than the entire screen.

        So: sizes are computed in unscaled units, positions in screen units.
        """
        try:
            scaling = ctk.ScalingTracker.get_window_scaling(self.root) or 1.0
        except Exception:
            scaling = 1.0

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        # Neither the taskbar nor the title bar is part of the geometry Tk
        # reports, so hold back enough room for both.
        usable_w = screen_w - 40
        usable_h = screen_h - 120

        width = int(min(1280, usable_w / scaling))
        height = int(min(860, usable_h / scaling))

        x = max(0, int((screen_w - width * scaling) / 2))
        y = max(0, int((usable_h - height * scaling) / 2))

        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(900, width), min(560, height))

    def on_closing(self):
        if self.processing:
            if not messagebox.askokcancel(
                "Quit",
                "An analysis is still running.\n\n"
                "Quitting now leaves a partial output folder behind. Quit anyway?",
            ):
                return
            self.processing = False
        self.force_quit()

    def force_quit(self):
        try:
            self.save_config()
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        # Worker threads are daemons, but FSCT's own thread pools are not
        # always, so exit hard rather than hang on a half-finished run.
        os._exit(0)

    # -- layout -------------------------------------------------------------
    def build_ui(self):
        self.root.configure(fg_color=APP_BG)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.build_sidebar()
        self.build_header()

        self.content = ctk.CTkFrame(self.root, fg_color="transparent")
        self.content.grid(row=1, column=1, sticky="nsew", padx=(0, 22), pady=(0, 6))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.build_page_point_cloud()
        self.build_page_tools()
        self.build_page_analysis()
        self.build_page_results()
        self.build_page_settings()
        self.build_status_bar()

        self._restyle_native()

    def build_sidebar(self):
        bar = ctk.CTkFrame(self.root, width=232, corner_radius=0, fg_color=SIDEBAR_BG)
        bar.grid(row=0, column=0, rowspan=3, sticky="nsw")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(0, weight=1)
        bar.grid_rowconfigure(2, weight=1)

        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=20, pady=(24, 22))
        brand.grid_columnconfigure(1, weight=1)

        logo = self.load_logo(46)
        if logo is not None:
            ctk.CTkLabel(brand, text="", image=logo).grid(row=0, column=0, rowspan=2, padx=(0, 12))

        ctk.CTkLabel(
            brand, text="FSCT", font=FONT_TITLE, text_color=TEXT, anchor="w"
        ).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(
            brand, text="Forest Structural\nComplexity Tool", font=FONT_TINY,
            text_color=TEXT_MUTED, anchor="w", justify="left",
        ).grid(row=1, column=1, sticky="w")

        nav = ctk.CTkFrame(bar, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", padx=14)
        nav.grid_columnconfigure(0, weight=1)

        for index, (key, label, _desc) in enumerate(self.PAGES):
            button = ctk.CTkButton(
                nav, text=f"  {label}", anchor="w", height=40, corner_radius=8,
                font=FONT_BODY, fg_color="transparent", hover_color=ACCENT_SOFT,
                text_color=TEXT, command=lambda k=key: self.show_page(k),
            )
            button.grid(row=index, column=0, sticky="ew", pady=3)
            self.nav_buttons[key] = button

        footer = ctk.CTkFrame(bar, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 20))
        footer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            footer, text="APPEARANCE", font=FONT_TINY, text_color=TEXT_MUTED, anchor="w"
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ctk.CTkSegmentedButton(
            footer, values=["Light", "Dark", "System"], variable=self.appearance_mode,
            command=self.set_appearance, font=FONT_TINY, height=30,
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
        ).grid(row=1, column=0, sticky="ew")

        ctk.CTkLabel(
            footer, text=f"Version {__version__}", font=FONT_TINY, text_color=TEXT_MUTED, anchor="w"
        ).grid(row=2, column=0, sticky="ew", pady=(14, 0))

    def load_logo(self, size):
        png = os.path.join(PROJECT_ROOT, 'icon.png')
        if not os.path.exists(png):
            return None
        try:
            from PIL import Image
            image = Image.open(png)
            self._logo_image = ctk.CTkImage(
                light_image=image, dark_image=image, size=(size, size)
            )
            return self._logo_image
        except Exception:
            return None

    def build_header(self):
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.grid(row=0, column=1, sticky="ew", padx=(24, 22), pady=(22, 14))
        header.grid_columnconfigure(0, weight=1)

        titles = ctk.CTkFrame(header, fg_color="transparent")
        titles.grid(row=0, column=0, sticky="w")

        self.page_title = ctk.CTkLabel(
            titles, text="", font=FONT_H1, text_color=TEXT, anchor="w"
        )
        self.page_title.grid(row=0, column=0, sticky="w")

        self.page_subtitle = ctk.CTkLabel(
            titles, text="", font=FONT_SMALL, text_color=TEXT_MUTED, anchor="w"
        )
        self.page_subtitle.grid(row=1, column=0, sticky="w")

        # Read-only: names the loaded file on the pages that have no other
        # reference to it. Hidden on the Point Cloud page, where the full
        # picker sits a few pixels below - see show_page. Deliberately has no
        # Open button: choosing a file belongs on the Point Cloud page, and a
        # second entry point here made it possible to swap the file from the
        # middle of configuring a run.
        self.file_chip_frame = ctk.CTkFrame(
            header, corner_radius=8, fg_color=CARD_BG,
            border_width=1, border_color=CARD_BORDER,
        )
        self.file_chip_frame.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            self.file_chip_frame, text="FILE", font=FONT_TINY, text_color=TEXT_MUTED
        ).grid(row=0, column=0, padx=(14, 8), pady=10)

        self.file_chip = ctk.CTkLabel(
            self.file_chip_frame, text="none selected", font=FONT_BODY,
            text_color=TEXT, anchor="w",
        )
        self.file_chip.grid(row=0, column=1, padx=(0, 16), pady=10)

    def build_status_bar(self):
        bar = ctk.CTkFrame(self.root, height=34, corner_radius=0, fg_color=SIDEBAR_BG)
        bar.grid(row=2, column=1, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            bar, text="Ready", font=FONT_SMALL, text_color=TEXT_MUTED, anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=16, pady=6)

        self.time_label = ctk.CTkLabel(
            bar, text="", font=FONT_SMALL, text_color=TEXT_MUTED, anchor="e"
        )
        self.time_label.grid(row=0, column=1, sticky="e", padx=16, pady=6)

    def page_frame(self, key):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_remove()
        self.pages[key] = frame
        return frame

    def show_page(self, key):
        for name, frame in self.pages.items():
            if name == key:
                frame.grid()
            else:
                frame.grid_remove()

        for name, button in self.nav_buttons.items():
            selected = name == key
            button.configure(
                fg_color=ACCENT if selected else "transparent",
                text_color="#FFFFFF" if selected else TEXT,
                hover_color=ACCENT_HOVER if selected else ACCENT_SOFT,
            )

        for name, label, description in self.PAGES:
            if name == key:
                self.page_title.configure(text=label)
                self.page_subtitle.configure(text=description)
                break

        # The Point Cloud page owns the file picker, so the header chip would
        # only repeat it there.
        if key == "point_cloud":
            self.file_chip_frame.grid_remove()
        else:
            self.file_chip_frame.grid()

        if key == "results":
            self.refresh_results()

    # -- page: point cloud --------------------------------------------------
    def build_page_point_cloud(self):
        page = self.page_frame("point_cloud")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)

        picker = Card(
            page,
            title="Input point cloud",
            description="FSCT reads uncompressed LAS. LAZ files are converted "
                        "automatically when you start an analysis.",
        )
        picker.grid(row=0, column=0, sticky="ew")
        picker.body.grid_columnconfigure(0, weight=1)

        entry = ctk.CTkEntry(
            picker.body, textvariable=self.current_file, height=38,
            font=FONT_BODY, placeholder_text="No file selected",
        )
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        entry.configure(state="readonly")

        accent_button(picker.body, "Browse...", self.browse_file, width=120).grid(
            row=0, column=1
        )
        ghost_button(picker.body, "Clear", self.clear_file, width=90).grid(
            row=0, column=2, padx=(10, 0)
        )

        tiles = ctk.CTkFrame(page, fg_color="transparent")
        tiles.grid(row=1, column=0, sticky="ew", pady=(16, 0))

        captions = [
            ("points", "Points"),
            ("density", "Points per m2"),
            ("extent", "Extent (X x Y)"),
            ("height", "Height range"),
            ("size", "File size"),
        ]
        for index, (key, caption) in enumerate(captions):
            tile = StatTile(tiles, caption)
            tile.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 12, 0))
            tiles.grid_columnconfigure(index, weight=1)
            self.stat_tiles[key] = tile

        details = Card(page, title="Header details")
        details.grid(row=2, column=0, sticky="nsew", pady=(16, 0))

        # Sits level with the card's own title rather than opening a band of
        # empty space between the heading and the text.
        ghost_button(
            details, "Reload header", self.refresh_file_info, width=140, height=30
        ).grid(row=0, column=1, sticky="e", padx=(0, 18), pady=(14, 0))

        self.info_text = ctk.CTkTextbox(
            details.body, font=FONT_MONO, fg_color=INSET_BG, text_color=TEXT,
            corner_radius=8, wrap="none",
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        details.body.grid_rowconfigure(0, weight=1)
        details.body.grid_columnconfigure(0, weight=1)
        self._set_textbox(self.info_text, "Select a point cloud to see its header.")

    # -- page: tools --------------------------------------------------------
    def build_page_tools(self):
        page = self.page_frame("tools")
        page.grid_columnconfigure((0, 1, 2), weight=1, uniform="tools")
        page.grid_rowconfigure(1, weight=1)

        convert = Card(
            page, title="Format conversion",
            description="LAZ is compressed and about a fifth of the size; FSCT "
                        "itself only reads LAS.",
        )
        convert.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        accent_button(convert.body, "LAZ  ->  LAS   (decompress)", self.convert_laz_to_las).grid(
            row=0, column=0, sticky="ew", pady=(0, 8)
        )
        ghost_button(convert.body, "LAS  ->  LAZ   (compress)", self.convert_las_to_laz).grid(
            row=1, column=0, sticky="ew"
        )

        resample = Card(
            page, title="Resample",
            description="Keeps one point per grid cell. Halving the density "
                        "roughly halves the segmentation time.",
        )
        resample.grid(row=0, column=1, sticky="nsew", padx=8)
        self.resample_step = tk.DoubleVar(value=0.05)
        ParamRow(
            resample.body, 0, "Grid step", self.resample_step, 0.01, 1.0, 0.01, "m"
        )
        accent_button(resample.body, "Resample", self.resample_cloud).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(14, 0)
        )

        viewer = Card(
            page, title="View and locate",
            description="LAStools' lasview renders the raw cloud in 3D. Save a "
                        "clean copy if it refuses to open the file - that keeps "
                        "every point.",
        )
        viewer.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        accent_button(viewer.body, "Open in lasview", self.open_lastools).grid(
            row=0, column=0, sticky="ew", pady=(0, 8)
        )
        ghost_button(viewer.body, "Save a clean copy", self.save_clean_copy).grid(
            row=1, column=0, sticky="ew", pady=(0, 8)
        )
        ghost_button(viewer.body, "Open containing folder", self.open_output_folder).grid(
            row=2, column=0, sticky="ew"
        )

        log_card = Card(page, title="Operation log")
        log_card.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(16, 0))
        self.operation_log = ctk.CTkTextbox(
            log_card.body, font=FONT_MONO, fg_color=INSET_BG, text_color=TEXT,
            corner_radius=8,
        )
        self.operation_log.grid(row=0, column=0, sticky="nsew")
        log_card.body.grid_rowconfigure(0, weight=1)
        self.operation_log.configure(state="disabled")

    # -- page: analysis -----------------------------------------------------
    def build_page_analysis(self):
        page = self.page_frame("analysis")
        page.grid_columnconfigure(0, weight=0, minsize=400)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        params = ctk.CTkScrollableFrame(
            page, fg_color=CARD_BG, corner_radius=12, width=390,
            label_text="  Parameters", label_font=FONT_H2, label_anchor="w",
            label_fg_color=CARD_BG, label_text_color=TEXT,
        )
        params.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        params.grid_columnconfigure(0, weight=1)

        # A conservative default that is safe on any card. Probing the GPU
        # means importing torch, which is far too slow to do while building the
        # window, so the warm-up worker revises this (see _apply_warm_up).
        self.batch_size = tk.IntVar(value=2)
        self.use_cpu = tk.BooleanVar(value=False)
        self.num_cores = tk.IntVar(value=0)
        self.plot_radius = tk.DoubleVar(value=0.0)
        self.plot_buffer = tk.DoubleVar(value=0.0)
        self.tree_cutoff = tk.DoubleVar(value=5.0)
        self.veg_cutoff = tk.DoubleVar(value=3.0)
        self.slice_thickness = tk.DoubleVar(value=0.15)
        self.slice_increment = tk.DoubleVar(value=0.05)
        self.height_percentile = tk.IntVar(value=100)
        self.sort_stems = tk.BooleanVar(value=True)
        self.generate_output = tk.BooleanVar(value=True)

        compute_rows = self._param_group(params, 0, "Compute", [
            # Bigger is not automatically faster here. Once the model's
            # activations no longer fit in VRAM the driver spills to host
            # memory and throughput collapses - on a 4 GB card batch 6 measured
            # 3x slower than batch 2.
            ("Batch size", self.batch_size, 1, 8, 1, "", "Checking the GPU..."),
            ("CPU cores", self.num_cores, 0, os.cpu_count() or 1, 1, "",
             "0 uses every core. Only affects the CPU-bound stages."),
        ], checkboxes=[("Force CPU only (ignore the GPU)", self.use_cpu)])
        self.batch_row = compute_rows["Batch size"]

        self._param_group(params, 1, "Plot", [
            ("Plot radius", self.plot_radius, 0, 100, 0.5, "m",
             "0 processes the whole cloud. Set a radius to crop a circular plot."),
            ("Plot buffer", self.plot_buffer, 0, 20, 0.5, "m",
             "Extra margin kept around the plot so edge trees stay complete."),
            ("Tree base height", self.tree_cutoff, 0, 10, 0.5, "m",
             "A stem must start below this height to count as a tree."),
            ("Ground veg height", self.veg_cutoff, 0, 10, 0.5, "m",
             "Vegetation below this is classified as understorey, not canopy."),
        ])

        self._param_group(params, 2, "Measurement", [
            ("Slice thickness", self.slice_thickness, 0.05, 0.30, 0.05, "m",
             "Thicker slices fit more stable cylinders but blur taper."),
            ("Slice increment", self.slice_increment, 0.01, 0.20, 0.01, "m",
             "Vertical spacing between fitted slices."),
            ("Height percentile", self.height_percentile, 90, 100, 1, "%",
             "Percentile used for tree height, to reject stray high returns."),
        ], checkboxes=[
            ("Sort stems into individual trees", self.sort_stems),
            ("Write the segmented output point cloud", self.generate_output),
        ])

        right = ctk.CTkFrame(page, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        controls = Card(right)
        controls.grid(row=0, column=0, sticky="ew")
        controls.body.grid_columnconfigure(0, weight=1)

        buttons = ctk.CTkFrame(controls.body, fg_color="transparent")
        buttons.grid(row=0, column=0, sticky="ew")
        buttons.grid_columnconfigure(0, weight=1)

        self.run_btn = accent_button(buttons, "Run FSCT analysis", self.run_inference, height=42)
        self.run_btn.grid(row=0, column=0, sticky="ew")

        self.stop_btn = ghost_button(buttons, "Stop", self.stop_inference, width=110, height=42)
        self.stop_btn.grid(row=0, column=1, padx=(10, 0))
        self.stop_btn.configure(state="disabled")

        # Determinate at 0 while idle. An idle indeterminate bar still paints
        # its moving chunk, which reads as "something is running".
        self.progress = ctk.CTkProgressBar(
            controls.body, mode="determinate", height=6, corner_radius=3,
            progress_color=ACCENT,
        )
        self.progress.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self.progress.set(0)

        status_row = ctk.CTkFrame(controls.body, fg_color="transparent")
        status_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        status_row.grid_columnconfigure(0, weight=1)

        self.stage_label = ctk.CTkLabel(
            status_row, text="Idle", font=FONT_SMALL, text_color=TEXT_MUTED, anchor="w"
        )
        self.stage_label.grid(row=0, column=0, sticky="w")

        self.elapsed_label = ctk.CTkLabel(
            status_row, text="", font=FONT_SMALL, text_color=TEXT_MUTED, anchor="e"
        )
        self.elapsed_label.grid(row=0, column=1, sticky="e")

        console_card = Card(right, title="Processing console")
        console_card.grid(row=1, column=0, sticky="nsew", pady=(16, 0))
        console_card.body.grid_rowconfigure(0, weight=1)

        self.console = ctk.CTkTextbox(
            console_card.body, font=FONT_MONO, corner_radius=8, wrap="none",
            fg_color=CONSOLE_BG, text_color=CONSOLE_FG,
        )
        self.console.grid(row=0, column=0, sticky="nsew")
        self.console.tag_config("ok", foreground="#5BD68F")
        self.console.tag_config("err", foreground="#FF8B72")
        self.console.tag_config("dim", foreground="#7C8A83")
        self.console.configure(state="disabled")

        actions = ctk.CTkFrame(console_card.body, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ghost_button(actions, "Clear console", self.clear_console, width=130, height=30).pack(side="left")
        ghost_button(actions, "Save log...", self.save_console, width=120, height=30).pack(side="left", padx=(10, 0))

    def _param_group(self, parent, index, title, rows, checkboxes=()):
        group = ctk.CTkFrame(parent, fg_color="transparent")
        group.grid(row=index, column=0, sticky="ew", padx=14, pady=(14, 0))
        group.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            group, text=title.upper(), font=FONT_TINY, text_color=ACCENT, anchor="w"
        ).grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))

        row = 1
        built = {}
        for label, variable, low, high, step, unit, hint in rows:
            built[label] = ParamRow(group, row, label, variable, low, high,
                                    step, unit, hint)
            row += ParamRow.rows_used(hint)

        for text, variable in checkboxes:
            ctk.CTkCheckBox(
                group, text=text, variable=variable, font=FONT_BODY,
                text_color=TEXT, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                checkbox_width=18, checkbox_height=18,
            ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(8, 0))
            row += 1

        ctk.CTkFrame(group, height=1, corner_radius=0, fg_color=CARD_BORDER).grid(
            row=row, column=0, columnspan=4, sticky="ew", pady=(16, 0)
        )
        return built

    # -- page: results ------------------------------------------------------
    def build_page_results(self):
        page = self.page_frame("results")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)

        tiles = ctk.CTkFrame(page, fg_color="transparent")
        tiles.grid(row=0, column=0, sticky="ew")

        captions = [
            ("trees", "Trees detected"),
            ("stems_ha", "Stems per hectare"),
            ("dbh", "Mean DBH"),
            ("height", "Mean height"),
            ("volume", "Stem volume"),
        ]
        for index, (key, caption) in enumerate(captions):
            tile = StatTile(tiles, caption)
            tile.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 12, 0))
            tiles.grid_columnconfigure(index, weight=1)
            self.result_tiles[key] = tile

        self.results_note = ctk.CTkLabel(
            page, text="No completed run yet.", font=FONT_SMALL,
            text_color=TEXT_MUTED, anchor="w",
        )
        self.results_note.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        files = Card(page, title="Output files")
        files.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        files.body.grid_rowconfigure(0, weight=1)
        files.body.grid_columnconfigure(0, weight=1)

        tree_container = ctk.CTkFrame(files.body, fg_color=INSET_BG, corner_radius=8)
        tree_container.grid(row=0, column=0, sticky="nsew")
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(
            tree_container, columns=('Type', 'Size', 'Modified'),
            show='tree headings', selectmode='browse', style="FSCT.Treeview",
        )
        self.results_tree.heading('#0', text='File')
        self.results_tree.heading('Type', text='Type')
        self.results_tree.heading('Size', text='Size')
        self.results_tree.heading('Modified', text='Modified')
        self.results_tree.column('#0', width=380, anchor='w')
        self.results_tree.column('Type', width=90, anchor='center')
        self.results_tree.column('Size', width=110, anchor='e')
        self.results_tree.column('Modified', width=170, anchor='center')
        self.results_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        self.results_tree.bind("<Double-1>", lambda _e: self.open_selected_result())

        scroll = ctk.CTkScrollbar(tree_container, command=self.results_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns", padx=(2, 6), pady=6)
        self.results_tree.configure(yscrollcommand=scroll.set)

        actions = ctk.CTkFrame(files.body, fg_color="transparent")
        actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ghost_button(actions, "Refresh", self.refresh_results, width=110).pack(side="left")
        ghost_button(actions, "Open selected", self.open_selected_result, width=140).pack(side="left", padx=(10, 0))
        ghost_button(actions, "Open output folder", self.open_output_folder, width=170).pack(side="left", padx=(10, 0))
        ghost_button(actions, "Plot summary table", self.view_plot_summary, width=170).pack(side="left", padx=(10, 0))

    # -- page: settings -----------------------------------------------------
    def build_page_settings(self):
        page = self.page_frame("settings")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        lastools = Card(
            page, title="LAStools",
            description="Optional. FSCT.bat downloads it automatically; it only "
                        "powers the external 3D viewer, everything else uses laspy.",
        )
        lastools.grid(row=0, column=0, sticky="ew")
        lastools.body.grid_columnconfigure(0, weight=1)

        ctk.CTkEntry(
            lastools.body, textvariable=self.lastools_path, height=36,
            font=FONT_BODY, placeholder_text="Path to the LAStools bin directory",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ghost_button(lastools.body, "Browse...", self.browse_lastools, width=110).grid(row=0, column=1)
        accent_button(lastools.body, "Save", self.save_config_interactive, width=90).grid(
            row=0, column=2, padx=(10, 0)
        )

        download = ctk.CTkFrame(lastools.body, fg_color="transparent")
        download.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ghost_button(download, "Download LAStools automatically", self.download_lastools, width=280).pack(side="left")
        ctk.CTkLabel(
            download, text="about 60 MB, unpacked into third_party\\LAStools",
            font=FONT_TINY, text_color=TEXT_MUTED,
        ).pack(side="left", padx=12)

        about = Card(page, title="About")
        about.grid(row=1, column=0, sticky="nsew", pady=(16, 0))
        about.body.grid_columnconfigure(0, weight=1)

        about_text = (
            "The Forest Structural Complexity Tool segments terrestrial LiDAR "
            "point clouds with a deep learning model and measures the trees it "
            "finds: stem diameter, height, volume and taper.\n\n"
            "Original FSCT by Sean Krisanski. This desktop interface wraps the "
            "same pipeline that scripts/run_tools.py drives."
        )
        ctk.CTkLabel(
            about.body, text=about_text, font=FONT_BODY, text_color=TEXT_MUTED,
            justify="left", anchor="w", wraplength=720,
        ).grid(row=0, column=0, sticky="ew")

        # Filled in by _apply_warm_up; describing PyTorch means importing it,
        # which is exactly the work being kept off the startup path.
        self.env_label = ctk.CTkLabel(
            about.body,
            text=f"Python {sys.version.split()[0]}\nFSCT modules: loading...",
            font=FONT_MONO, text_color=TEXT_MUTED, justify="left", anchor="w",
        )
        self.env_label.grid(row=1, column=0, sticky="ew", pady=(16, 0))

        links = ctk.CTkFrame(about.body, fg_color="transparent")
        links.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        ghost_button(
            links, "FSCT on GitHub",
            lambda: webbrowser.open('https://github.com/SKrisanski/FSCT'), width=170,
        ).pack(side="left")
        ghost_button(links, "Documentation", self.open_documentation, width=170).pack(
            side="left", padx=(10, 0)
        )

    @staticmethod
    def torch_description():
        try:
            import torch
        except Exception as error:
            return f"PyTorch: not importable ({error.__class__.__name__})"
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return f"PyTorch {torch.__version__} - CUDA on {name} ({total:.1f} GB)"
            except Exception:
                return f"PyTorch {torch.__version__} - CUDA available"
        return f"PyTorch {torch.__version__} - CPU only"

    # -- appearance ---------------------------------------------------------
    def set_appearance(self, mode=None):
        mode = mode or self.appearance_mode.get()
        self.appearance_mode.set(mode)
        ctk.set_appearance_mode(mode)
        self._restyle_native()
        self.save_config()

    def _restyle_native(self):
        """
        Recolour the widgets CustomTkinter does not own.

        ttk.Treeview is the only classic widget left, and ttk styles hold
        literal colours rather than (light, dark) pairs, so they have to be
        rewritten every time the appearance mode changes.
        """
        style = ttk.Style()
        try:
            style.theme_use("clam")  # the only built-in theme that honours field colours
        except tk.TclError:
            pass

        background = shade(INSET_BG)
        foreground = shade(TEXT)
        style.configure(
            "FSCT.Treeview",
            background=background,
            fieldbackground=background,
            foreground=foreground,
            borderwidth=0,
            rowheight=26,
            font=("Segoe UI", 10),
        )
        style.configure(
            "FSCT.Treeview.Heading",
            background=shade(CARD_BG),
            foreground=shade(TEXT_MUTED),
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "FSCT.Treeview",
            background=[("selected", shade(ACCENT))],
            foreground=[("selected", "#FFFFFF")],
        )
        style.map("FSCT.Treeview.Heading", background=[("active", shade(ACCENT_SOFT))])

    # -- batch size ---------------------------------------------------------
    @staticmethod
    def recommended_batch_size():
        """
        Pick a default batch size that fits in VRAM, and describe why.

        Segmentation memory scales with the batch, and exceeding VRAM is far
        worse than a small batch: measured on a 4 GB RTX 3050, fp32 batch 2
        took 17 s, batch 4 took 25 s and batch 6 took 54 s, because the driver
        was paging activations to host memory. Mixed precision roughly halves
        the requirement, which is why the thresholds below are generous.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return 2, "No CUDA GPU detected - segmentation runs on the CPU and will be slow."
            name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            return 2, "Could not query the GPU; using a conservative batch size of 2."

        if total_gb >= 16:
            batch = 8
        elif total_gb >= 10:
            batch = 6
        else:
            batch = 4  # ~1.6 GB in mixed precision, comfortable on a 4 GB card

        return batch, (
            f"{name} ({total_gb:.1f} GB) - suggested {batch}. Raising this past "
            "what VRAM holds makes it dramatically slower, not faster."
        )

    # -- file handling ------------------------------------------------------
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select point cloud",
            filetypes=[
                ("Point clouds", "*.las *.laz"),
                ("LAS files", "*.las"),
                ("LAZ files", "*.laz"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.current_file.set(filename)
            self.refresh_file_info()
            self.update_status(f"Loaded {os.path.basename(filename)}")

    def _browse_from_point_cloud(self):
        """Show the page that owns file selection, then open the picker."""
        self.show_page("point_cloud")
        self.browse_file()

    def clear_file(self):
        self.current_file.set("")
        self.output_directory.set("")
        self._file_header = None
        self._set_textbox(self.info_text, "Select a point cloud to see its header.")
        for tile in self.stat_tiles.values():
            tile.set("--")
        self._update_file_chip()
        self.update_status("File cleared")

    def _update_file_chip(self):
        path = self.current_file.get()
        self.file_chip.configure(text=os.path.basename(path) if path else "none selected")

    def refresh_file_info(self):
        """
        Show the header without loading the points.

        laspy.read() pulls every point into memory - on a multi-gigabyte scan
        that froze the window for a minute just to display a bounding box.
        laspy.open() reads the header only and returns immediately.
        """
        self._update_file_chip()
        path = self.current_file.get()
        if not path or not os.path.exists(path):
            return

        try:
            import laspy
            with laspy.open(path) as reader:
                header = reader.header
        except Exception as error:
            self._file_header = None
            for tile in self.stat_tiles.values():
                tile.set("--")
            self._set_textbox(self.info_text, f"Could not read this file:\n\n{error}")
            return

        size_mb = os.path.getsize(path) / (1024 * 1024)
        self._file_header = (path, header, size_mb)

        mins, maxs = list(header.mins), list(header.maxs)
        # Plenty of scanners and exporters leave the header's min/max fields at
        # zero - data/test/example.las is one of them, and it is why the extent
        # used to read "0.00 to 0.00". Fall back to scanning the coordinates.
        if any(maxs[axis] <= mins[axis] for axis in range(3)):
            self._render_file_details(None, None, "scanning")
            self._start_worker(lambda: self._scan_extent(path))
        else:
            self._render_file_details(mins, maxs, "header")

    def _render_file_details(self, mins, maxs, source):
        """
        Fill the tiles and the header panel.

        `source` is "header" when the bounds came from the LAS header,
        "scanned" when they were measured from the points, and "scanning"
        while that measurement is still running.
        """
        if not self._file_header:
            return
        path, header, size_mb = self._file_header
        count = header.point_count

        self.stat_tiles["points"].set(self._compact_number(count))
        self.stat_tiles["size"].set(f"{size_mb:,.1f} MB")

        if mins is None or maxs is None:
            placeholder = "reading..." if source == "scanning" else "--"
            for key in ("density", "extent", "height"):
                self.stat_tiles[key].set(placeholder)
            extent_block = (
                "X             not recorded in the header\n"
                "Y             not recorded in the header\n"
                "Z             not recorded in the header\n"
                "\n"
                + ("Footprint     measuring from the points...\n"
                   if source == "scanning" else
                   "Footprint     unavailable\n")
            )
        else:
            span = [maxs[i] - mins[i] for i in range(3)]
            area = span[0] * span[1]
            self.stat_tiles["density"].set(f"{count / area:,.0f}" if area > 0 else "--")
            # A single-tree or transect scan is only metres across, and "14 x 2"
            # loses the shape of it; a full plot does not need the decimals.
            fmt = "{:.0f}" if max(span[0], span[1]) >= 100 else "{:.1f}"
            self.stat_tiles["extent"].set(
                f"{fmt.format(span[0])} x {fmt.format(span[1])} m"
            )
            self.stat_tiles["height"].set(f"{span[2]:.1f} m")
            note = "" if source == "header" else "   (measured from the points)"
            extent_block = (
                f"X             {mins[0]:.2f} to {maxs[0]:.2f}   span {span[0]:.2f} m\n"
                f"Y             {mins[1]:.2f} to {maxs[1]:.2f}   span {span[1]:.2f} m\n"
                f"Z             {mins[2]:.2f} to {maxs[2]:.2f}   span {span[2]:.2f} m{note}\n"
                f"\n"
                f"Footprint     {area:,.0f} m2 ({area / 10000:.3f} ha)\n"
            )

        self._set_textbox(self.info_text, (
            f"File          {os.path.basename(path)}\n"
            f"Path          {path}\n"
            f"\n"
            f"Points        {count:,}\n"
            f"Point format  {header.point_format.id}\n"
            f"LAS version   {header.version.major}.{header.version.minor}\n"
            # float() around each value on purpose: header.scales is a numpy
            # array, and numpy 2 renders its scalars as "np.float64(0.001)".
            f"Scales        {', '.join(f'{float(s):g}' for s in header.scales)}\n"
            f"Offsets       {', '.join(f'{float(o):.3f}' for o in header.offsets)}\n"
            f"\n"
            + extent_block +
            f"File size     {size_mb:,.2f} MB\n"
        ))

    def _scan_extent(self, path):
        """
        Measure the real bounding box for a file whose header does not have one.

        Streamed in chunks rather than with laspy.read(), so a multi-gigabyte
        scan costs a pass over the file instead of its full size in RAM.
        """
        try:
            import laspy
            import numpy as np

            low = np.full(3, np.inf)
            high = np.full(3, -np.inf)
            with laspy.open(path) as reader:
                for chunk in reader.chunk_iterator(2_000_000):
                    points = np.vstack((chunk.x, chunk.y, chunk.z)).T
                    if not len(points):
                        continue
                    low = np.minimum(low, points.min(axis=0))
                    high = np.maximum(high, points.max(axis=0))

            if not np.isfinite(low).all():
                raise ValueError("the file contains no points")
            result = (low.tolist(), high.tolist())
        except Exception as error:
            self.log_operation(f"Could not measure the extent of "
                               f"{os.path.basename(path)}: {error}")
            result = None

        def apply():
            # The user may have loaded a different file while this ran.
            if self.current_file.get() != path:
                return
            if result is None:
                self._render_file_details(None, None, "failed")
            else:
                self._render_file_details(result[0], result[1], "scanned")

        self._on_main_thread(apply)

    @staticmethod
    def _compact_number(value):
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return f"{value:,}"

    def _set_textbox(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_current_file(self, path):
        """Set the current file from any thread and refresh the info panel."""
        def apply():
            self.current_file.set(path)
            self.refresh_file_info()
        self._on_main_thread(apply)

    def _start_worker(self, target):
        """Start a daemon worker thread and keep a reference to it."""
        thread = threading.Thread(target=target, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    # -- conversions --------------------------------------------------------
    def convert_laz_to_las(self, on_success=None):
        """
        Convert LAZ to LAS.

        on_success, if given, is called on the main thread with the path of the
        converted file once the conversion has actually finished.
        """
        file_path = self.current_file.get()
        if not file_path or not file_path.lower().endswith('.laz'):
            messagebox.showwarning("Nothing to convert", "Select a LAZ file first.")
            return

        self.log_operation("Converting LAZ to LAS...")
        # Change only the extension. str.replace would also rewrite any earlier
        # ".laz" in the path, e.g. C:\laz_data\scan.laz -> C:\las_data\scan.las.
        output_file = os.path.splitext(file_path)[0] + '.las'

        def convert():
            try:
                import laspy
                las = laspy.read(file_path)
                las.write(output_file)
                self._set_current_file(output_file)
                self.log_operation(f"Converted to {os.path.basename(output_file)}")
                self.update_status("Conversion complete")
                if on_success:
                    self._on_main_thread(on_success, output_file)
            except Exception as e:
                error_msg = str(e)
                self.log_operation(f"Error: {error_msg}")
                self._on_main_thread(
                    lambda msg=error_msg: messagebox.showerror("Error", f"Conversion failed:\n{msg}")
                )

        self._start_worker(convert)

    def convert_las_to_laz(self):
        file_path = self.current_file.get()
        if not file_path or not file_path.lower().endswith('.las'):
            messagebox.showwarning("Nothing to convert", "Select a LAS file first.")
            return

        self.log_operation("Compressing LAS to LAZ...")
        output_file = os.path.splitext(file_path)[0] + '.laz'

        def convert():
            try:
                import laspy
                las = laspy.read(file_path)
                las.write(output_file)
                self._set_current_file(output_file)
                self.log_operation(f"Compressed to {os.path.basename(output_file)}")
                self.update_status("Compression complete")
            except Exception as e:
                error_msg = str(e)
                if "LazBackend" in error_msg or "lazrs" in error_msg.lower():
                    error_msg = (
                        "LAZ compression is not available. Install the lazrs package:\n\n"
                        "    pip install lazrs\n\n"
                        "Or run FSCT.bat setup /force to rebuild the environment."
                    )
                self.log_operation(f"Error: {error_msg}")
                self._on_main_thread(
                    lambda msg=error_msg: messagebox.showerror("Error", f"Compression failed:\n\n{msg}")
                )

        self._start_worker(convert)

    def resample_cloud(self):
        file_path = self.current_file.get()
        if not file_path:
            messagebox.showwarning("No file", "Select a point cloud first.")
            return

        step_size = self.resample_step.get()
        if step_size <= 0:
            messagebox.showwarning("Invalid step", "The grid step must be greater than zero.")
            return

        self.log_operation(f"Resampling with a {step_size} m grid step...")

        stem, file_ext = os.path.splitext(file_path)
        output_file = f'{stem}_resampled{file_ext}'

        def resample():
            try:
                import laspy
                import numpy as np

                las = laspy.read(file_path)
                xyz = np.vstack((las.x, las.y, las.z)).T

                # Grid-based resampling: keep one point per occupied cell.
                min_coords = xyz.min(axis=0)
                max_coords = xyz.max(axis=0)

                # int64 throughout. The default int is int32 on Windows, and a
                # flattened 3D cell id overflows it easily - a 50 m cube at a
                # 5 cm step is already 1e9 cells - which silently aliases
                # distant points onto the same id and drops real points.
                n_cells = np.floor((max_coords - min_coords) / step_size).astype(np.int64) + 1
                cell_indices = np.floor((xyz - min_coords) / step_size).astype(np.int64)
                cell_ids = (cell_indices[:, 0] +
                            cell_indices[:, 1] * n_cells[0] +
                            cell_indices[:, 2] * n_cells[0] * n_cells[1])

                _, unique_indices = np.unique(cell_ids, return_index=True)

                # Build a fresh header so the point count is recomputed, but
                # keep the source scales/offsets so coordinates round-trip.
                new_header = laspy.LasHeader(
                    point_format=las.header.point_format,
                    version=str(las.header.version),
                )
                new_header.offsets = las.header.offsets
                new_header.scales = las.header.scales

                resampled_las = laspy.LasData(new_header)
                resampled_las.points = las.points[unique_indices]
                resampled_las.write(output_file)

                reduction = (1 - len(unique_indices) / len(las.points)) * 100
                self.log_operation(
                    f"Resampled to {len(unique_indices):,} points ({reduction:.1f}% reduction)"
                )
                self.log_operation(f"Output: {os.path.basename(output_file)}")
                self._set_current_file(output_file)
                self.update_status("Resampling complete")

            except Exception as e:
                import traceback
                error_msg = str(e)
                self.log_operation(f"Error: {error_msg}")
                self.log_operation(traceback.format_exc())
                self._on_main_thread(
                    lambda msg=error_msg: messagebox.showerror("Error", f"Resampling failed:\n{msg}")
                )

        self._start_worker(resample)

    def save_clean_copy(self, file_path=None):
        """
        Rewrite a file through laspy, keeping every point.

        Defaults to the selected file, which is what the Tools button wants;
        open_in_lasview passes an explicit path so it can repair an output
        cloud from the Results list.

        This is a repair, not a resample. laspy reads the points and writes
        them to a fresh header, which carries the georeferencing VLR across but
        leaves behind the stale "laszip encoded" and extra-bytes records that
        stop LAStools opening the file. Verified on a 5,104,837 point cloud:
        identical coordinates, intensity, GPS time and classification, same
        scales, offsets, version and point format - only the dead VLRs go.
        """
        if file_path is None:
            file_path = self.current_file.get()
        if not file_path:
            messagebox.showwarning("No file", "Select a point cloud first.")
            return

        stem, extension = os.path.splitext(file_path)
        output_file = f"{stem}_clean{extension}"

        if os.path.exists(output_file) and not messagebox.askyesno(
            "Overwrite?",
            f"{os.path.basename(output_file)} already exists.\n\nReplace it?",
        ):
            return

        self.log_operation(f"Writing a clean copy of {os.path.basename(file_path)}...")
        self.update_status("Writing a clean copy...")

        def repair():
            try:
                import laspy
                las = laspy.read(file_path)
                las.write(output_file)

                remaining = laszip_vlr_mismatch(output_file)
                if remaining:
                    raise ValueError(
                        "the rewritten file still disagrees with itself "
                        f"({remaining[0]} vs {remaining[1]} bytes)"
                    )

                self.log_operation(
                    f"Clean copy written: {os.path.basename(output_file)} "
                    f"({len(las.points):,} points, none discarded)"
                )
                self._set_current_file(output_file)
                self.update_status("Clean copy ready")
                self._on_main_thread(
                    lambda: messagebox.showinfo(
                        "Clean copy ready",
                        f"Saved as:\n{output_file}\n\n"
                        "It is now the selected file, so 'Open in lasview' will "
                        "use it.",
                    )
                )
            except Exception as e:
                error_msg = str(e)
                self.log_operation(f"Error: {error_msg}")
                self.update_status("Could not write a clean copy")
                self._on_main_thread(
                    lambda msg=error_msg: messagebox.showerror(
                        "Error", f"Could not write a clean copy:\n{msg}"
                    )
                )

        self._start_worker(repair)

    def open_lastools(self):
        """Show the file selected on the Point Cloud page in lasview."""
        file_path = self.current_file.get()
        if not file_path:
            messagebox.showwarning("No file", "Select a point cloud first.")
            return
        self.open_in_lasview(file_path)

    def open_in_lasview(self, file_path):
        """
        Show any point cloud in lasview, not only the currently selected one.

        Split out from open_lastools so the Results list can hand it one of the
        run's output clouds without disturbing the file chosen for processing.
        """
        self.autodetect_lastools()
        lastools = self.lastools_path.get()
        if not lastools:
            if messagebox.askyesno(
                "LAStools not installed",
                "LAStools is not set up yet.\n\nDownload it now? (about 60 MB)",
            ):
                self.download_lastools()
            return

        lasview_exe = os.path.join(lastools, 'lasview.exe')
        if not os.path.exists(lasview_exe):
            messagebox.showerror("Not found", f"lasview.exe is not at:\n{lasview_exe}")
            return

        # lasview opens its window before it reads the file and leaves it blank
        # when the read fails, writing the reason to a console the user may not
        # have. Catch the one inconsistency that actually stops it, up front.
        mismatch = laszip_vlr_mismatch(file_path)
        if mismatch:
            declared, actual = mismatch
            self.log_operation(
                f"lasview refused {os.path.basename(file_path)}: stale LASzip VLR "
                f"describes {declared}-byte points, header says {actual}"
            )
            if messagebox.askyesno(
                "lasview cannot read this file",
                f"{os.path.basename(file_path)} contains a leftover LASzip "
                "record from an earlier format conversion.\n\n"
                f"It describes {declared}-byte points, but the file's own header "
                f"says {actual} bytes. LAStools trusts that record and refuses "
                "to open the file - it will suggest upgrading LAStools, which "
                "does not help, because the file is what is inconsistent.\n\n"
                "FSCT itself is unaffected: laspy ignores the stale record, so "
                "processing this file works normally.\n\n"
                "Save a clean copy now? Every point is kept - the copy only "
                "drops the stale record, so it is not a resample and nothing is "
                "thinned.",
            ):
                # Repair the file we were actually asked to show, which is not
                # necessarily the one selected for processing.
                self.save_clean_copy(file_path)
            return

        try:
            process = subprocess.Popen(
                [lasview_exe, file_path],
                stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch LAStools:\n{e}")
            return

        self.update_status("Launched the LAStools viewer")
        self._start_worker(lambda: self._watch_lasview(process, file_path))

    def _watch_lasview(self, process, file_path):
        """
        Report anything lasview complains about instead of leaving a blank window.

        lasview keeps running with an empty viewport when it cannot read a file,
        so waiting for a non-zero exit code would never fire. Read whatever it
        writes to stderr and surface the error lines.
        """
        try:
            stderr = process.stderr
            lines = []
            for line in iter(stderr.readline, ''):
                lines.append(line.rstrip())
                if len(lines) > 40:
                    break
            stderr.close()
        except Exception:
            return

        errors = [line for line in lines if line.startswith("ERROR")]
        if not errors:
            return

        for line in errors:
            self.log_operation(f"lasview: {line}")

        detail = "\n".join(errors[:6])
        self._on_main_thread(
            lambda: messagebox.showerror(
                "lasview could not display this file",
                f"{os.path.basename(file_path)} opened an empty viewer window.\n\n"
                f"{detail}\n\n"
                "See the operation log on the Tools page for the full output.",
            )
        )

    def open_output_folder(self):
        file_path = self.current_file.get()
        if not file_path:
            messagebox.showwarning("No file", "Select a point cloud first.")
            return

        output_dir = os.path.splitext(file_path)[0] + "_FSCT_output"
        target = output_dir if os.path.isdir(output_dir) else os.path.dirname(
            os.path.abspath(file_path)
        )
        try:
            os.startfile(target)
        except OSError as e:
            messagebox.showerror("Error", f"Could not open the folder:\n{e}")

    # -- inference ----------------------------------------------------------
    def run_inference(self):
        file_path = self.current_file.get()
        if not file_path:
            messagebox.showwarning("No file", "Select a point cloud first.")
            return

        if self.fsct_error is not None:
            # Surface the actual import error. "Please check installation" on
            # its own gives the user nothing to act on, and the usual cause is
            # a specific missing package (torch_geometric, mdutils, ...).
            messagebox.showerror(
                "FSCT modules unavailable",
                "The FSCT modules could not be imported:\n\n"
                f"{self.fsct_error}\n\n"
                "Re-run FSCT.bat setup /force to rebuild the environment.",
            )
            return

        if self.processing:
            messagebox.showinfo("Already running", "An analysis is already in progress.")
            return

        # FSCT reads .las only. Convert first, then continue from the
        # conversion's completion callback - the previous version retried on a
        # fixed 2 s timer, which raced the conversion on anything but a small
        # file and re-ran with the original .laz path.
        if file_path.lower().endswith('.laz'):
            if messagebox.askyesno(
                "Convert first?",
                "FSCT reads uncompressed LAS. Convert this LAZ file and continue?",
            ):
                self.convert_laz_to_las(on_success=lambda _path: self.run_inference())
            return

        self.processing = True
        self._start_time = time.time()
        self._set_stage("Starting")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

        self.console_log("=" * 62)
        self.console_log("FSCT analysis", "ok")
        self.console_log("=" * 62)
        self.console_log(f"File : {file_path}")
        self.console_log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.console_log("")

        parameters = {
            'point_cloud_filename': file_path,
            'plot_centre': None,
            'plot_radius': self.plot_radius.get(),
            'plot_radius_buffer': self.plot_buffer.get(),
            'batch_size': self.batch_size.get(),
            'num_cpu_cores': self.num_cores.get() if self.num_cores.get() > 0 else os.cpu_count(),
            'use_CPU_only': self.use_cpu.get(),
            'slice_thickness': self.slice_thickness.get(),
            'slice_increment': self.slice_increment.get(),
            'sort_stems': 1 if self.sort_stems.get() else 0,
            'height_percentile': self.height_percentile.get(),
            'tree_base_cutoff_height': self.tree_cutoff.get(),
            'generate_output_point_cloud': 1 if self.generate_output.get() else 0,
            'ground_veg_cutoff_height': self.veg_cutoff.get(),
            'veg_sorting_range': 1.5,
            'stem_sorting_range': 1.0,
            'taper_measurement_height_min': 0,
            'taper_measurement_height_max': 30,
            'taper_measurement_height_increment': 0.2,
            'taper_slice_thickness': 0.4,
            'delete_working_directory': True,
            'minimise_output_size_mode': 0,
        }

        output_dir = os.path.splitext(file_path)[0] + "_FSCT_output"
        self.output_directory.set(output_dir)

        def process():
            # Capture these before the try block. The old code assigned
            # old_stdout inside the try and restored it in the except handler,
            # so any failure before the assignment raised NameError and hid the
            # real error.
            old_stdout, old_stderr = sys.stdout, sys.stderr
            # No terminal mirror: the run is shown in the processing console,
            # and echoing it to the launching terminal as well just prints
            # every progress line twice.
            stream = ConsoleStream(self._console_sink)
            sys.stdout = stream
            sys.stderr = stream

            try:
                # Normally already imported by the warm-up; if the user got
                # here first this blocks the worker, not the window.
                if not self.fsct_ready:
                    self._set_stage("Loading FSCT modules")
                fsct, defaults = load_fsct_modules()
                parameters.update(defaults)

                fsct(
                    parameters=parameters,
                    preprocess=True,
                    segmentation=True,
                    postprocessing=True,
                    measure_plot=True,
                    make_report=True,
                    clean_up_files=False,
                )

                self.console_log("")
                self.console_log("=" * 62)
                self.console_log("Processing complete", "ok")
                self.console_log(f"Output directory: {output_dir}", "ok")
                self.console_log("=" * 62)
                self._on_main_thread(self.processing_complete, output_dir)

            except Exception as e:
                import traceback
                self.console_log("")
                self.console_log("=" * 62, "err")
                self.console_log("Processing failed", "err")
                self.console_log("=" * 62, "err")
                self.console_log(traceback.format_exc(), "err")
                self._on_main_thread(self.processing_failed, str(e))

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        self._start_worker(process)

    def stop_inference(self):
        """
        Explain why there is nothing to cancel.

        FSCT() is a single long-running call with no cancellation points, so
        there is nothing to interrupt mid-run. Be honest about that rather than
        claiming it will "halt at the next checkpoint", which it never did.
        """
        messagebox.showinfo(
            "Cannot stop mid-run",
            "FSCT runs as one uninterruptible job, so it cannot be cancelled "
            "once it has started.\n\n"
            "To abort, close this window - you will be asked to confirm, and "
            "the partial output folder can then be deleted.",
        )

    def _finish_processing(self, status):
        self.processing = False
        self._start_time = None
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)
        self.update_status(status)

    def processing_complete(self, output_dir):
        self._finish_processing("Processing complete")
        self._set_stage("Finished")
        self.show_page("results")  # show_page("results") refreshes the listing
        messagebox.showinfo(
            "Analysis complete",
            f"FSCT finished successfully.\n\nResults are in:\n{output_dir}",
        )

    def processing_failed(self, error):
        self._finish_processing("Processing failed")
        self._set_stage("Failed")
        messagebox.showerror("Analysis failed", f"Processing failed:\n\n{error}")

    def _set_stage(self, stage):
        """Record the current pipeline stage. Safe to call from any thread."""
        # Assigning a str is atomic, so the pump can pick it up without a lock
        # and no worker ever touches the label itself.
        self._pending_stage = stage

    # -- results ------------------------------------------------------------
    def refresh_results(self):
        output_dir = self.output_directory.get()
        if not output_dir or not os.path.isdir(output_dir):
            file_path = self.current_file.get()
            if file_path:
                output_dir = os.path.splitext(file_path)[0] + "_FSCT_output"
                self.output_directory.set(output_dir)

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if not output_dir or not os.path.isdir(output_dir):
            self.results_note.configure(text="No output folder for the current file yet.")
            for tile in self.result_tiles.values():
                tile.set("--")
            return

        total_bytes = 0
        count = 0
        for root, _dirs, files in os.walk(output_dir):
            for name in sorted(files):
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                    modified = datetime.fromtimestamp(os.path.getmtime(path))
                except OSError:
                    continue

                total_bytes += size
                count += 1
                size_str = (f"{size / 1024:.1f} KB" if size < 1024 * 1024
                            else f"{size / (1024 * 1024):.1f} MB")
                self.results_tree.insert(
                    '', 'end',
                    text=os.path.relpath(path, output_dir),
                    values=(
                        os.path.splitext(name)[1][1:].upper() or "File",
                        size_str,
                        modified.strftime('%Y-%m-%d %H:%M'),
                    ),
                )

        self.results_note.configure(
            text=f"{count} file(s), {total_bytes / (1024 * 1024):.1f} MB in {output_dir}"
        )
        self.update_summary(output_dir)

    def update_summary(self, output_dir):
        """Fill the headline tiles from plot_summary.csv."""
        for tile in self.result_tiles.values():
            tile.set("--")

        summary_file = os.path.join(output_dir, 'plot_summary.csv')
        if not os.path.exists(summary_file):
            return

        try:
            import pandas as pd
            df = pd.read_csv(summary_file)
        except Exception as error:
            self.results_note.configure(text=f"Could not read plot_summary.csv: {error}")
            return

        # Column names must match the headers written by
        # scripts/preprocessing.py. An earlier version looked for 'Number of
        # Trees' / 'Total Basal Area', which do not exist in plot_summary.csv,
        # so only 'Mean DBH' ever displayed.
        mapping = [
            ("trees", 'Num Trees in Plot', '{:,.0f}', ''),
            ("stems_ha", 'Stems/ha', '{:,.0f}', ''),
            ("dbh", 'Mean DBH', '{:.3f}', ' m'),
            ("height", 'Mean Height', '{:.1f}', ' m'),
            ("volume", 'Total Volume 1', '{:.2f}', ' m3'),
        ]
        for key, column, fmt, unit in mapping:
            if column in df.columns and len(df):
                value = df[column].values[0]
                if pd.notna(value):
                    self.result_tiles[key].set(fmt.format(value) + unit)

    def open_selected_result(self):
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Nothing selected", "Select a file in the list first.")
            return

        filename = self.results_tree.item(selection[0])['text']
        path = os.path.join(self.output_directory.get(), filename)
        if not os.path.exists(path):
            messagebox.showerror("Missing", f"That file is gone:\n{path}")
            return

        # Point clouds go to lasview. Windows has no default association for
        # .las/.laz, so os.startfile either did nothing or threw up the "how do
        # you want to open this file?" picker. Everything else - the report
        # figures, the CSVs, the markdown - is left to whatever the user has
        # associated, which is what they want for those.
        if os.path.splitext(path)[1].lower() in ('.las', '.laz'):
            self.open_in_lasview(path)
            return

        try:
            os.startfile(path)
        except OSError as e:
            messagebox.showerror("Error", f"Windows could not open the file:\n{e}")

    def view_plot_summary(self):
        output_dir = self.output_directory.get()
        summary_file = os.path.join(output_dir, 'plot_summary.csv') if output_dir else ''
        if not summary_file or not os.path.exists(summary_file):
            messagebox.showwarning("Not available", "plot_summary.csv has not been written yet.")
            return

        try:
            import pandas as pd
            df = pd.read_csv(summary_file)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read the plot summary:\n{e}")
            return

        popup = ctk.CTkToplevel(self.root)
        popup.title("Plot summary")
        popup.geometry("900x520")
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(0, weight=1)
        # A CTkToplevel takes a moment to map; setting the icon immediately is
        # ignored on Windows, so defer it.
        popup.after(220, lambda: self._set_toplevel_icon(popup))

        container = ctk.CTkFrame(popup, fg_color=INSET_BG, corner_radius=0)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # One row per metric reads far better than a single 40-column wide row
        # that needs horizontal scrolling to see anything.
        tree = ttk.Treeview(
            container, columns=("value",), show="tree headings", style="FSCT.Treeview"
        )
        tree.heading("#0", text="Metric")
        tree.heading("value", text="Value")
        tree.column("#0", width=340, anchor="w")
        tree.column("value", width=460, anchor="w")

        if len(df):
            row = df.iloc[0]
            for column in df.columns:
                tree.insert('', 'end', text=str(column), values=(str(row[column]),))

        tree.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        scroll = ctk.CTkScrollbar(container, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns", padx=(2, 8), pady=8)
        tree.configure(yscrollcommand=scroll.set)

    def _set_toplevel_icon(self, window):
        ico = os.path.join(PROJECT_ROOT, 'icon.ico')
        if os.path.exists(ico):
            try:
                window.iconbitmap(ico)
            except tk.TclError:
                pass

    # -- settings -----------------------------------------------------------
    def browse_lastools(self):
        directory = filedialog.askdirectory(title="Select the LAStools 'bin' directory")
        if directory:
            self.lastools_path.set(directory)

    def download_lastools(self):
        if not LASTOOLS_SETUP_AVAILABLE:
            messagebox.showerror("Error", "setup_lastools.py is not next to fsct_desktop.py.")
            return

        self.update_status("Downloading LAStools...")
        self.log_operation("Downloading LAStools (about 60 MB), please wait...")

        def worker():
            try:
                bin_dir = ensure_lastools(progress=None, quiet=True)
            except Exception as e:
                bin_dir = None
                self.log_operation(f"LAStools download failed: {e}")

            def done():
                if bin_dir:
                    self.lastools_path.set(bin_dir)
                    self.save_config()
                    self.update_status("LAStools ready")
                    self.log_operation(f"LAStools installed: {bin_dir}")
                    messagebox.showinfo("Done", f"LAStools is ready:\n\n{bin_dir}")
                else:
                    self.update_status("LAStools download failed")
                    messagebox.showerror(
                        "Download failed",
                        "Could not download LAStools.\n\n"
                        "Check your internet connection, or install it manually from\n"
                        "https://rapidlasso.de/lastools/ and set the path in Settings.",
                    )
            self._on_main_thread(done)

        self._start_worker(worker)

    def autodetect_lastools(self):
        """Pick up a LAStools copy already unpacked under third_party/."""
        if not LASTOOLS_SETUP_AVAILABLE or self.lastools_path.get():
            return
        try:
            bin_dir = find_lastools_bin()
        except Exception:
            bin_dir = None
        if bin_dir:
            self.lastools_path.set(bin_dir)

    def load_config(self):
        config_file = os.path.join(PROJECT_ROOT, 'gui_config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as handle:
                    config = json.load(handle)
                self.lastools_path.set(config.get('lastools_path', ''))
                mode = config.get('appearance_mode', 'System')
                if mode in ("Light", "Dark", "System"):
                    self.appearance_mode.set(mode)
            except (ValueError, OSError):
                pass

        # A config pointing at a path that no longer exists (a stale entry from
        # another machine, say) is worse than no config at all - clear it so
        # autodetection can take over.
        configured = self.lastools_path.get()
        if configured and not os.path.isdir(configured):
            self.lastools_path.set('')

        self.autodetect_lastools()

    def save_config(self):
        """Write settings to gui_config.json. Returns True on success."""
        config_file = os.path.join(PROJECT_ROOT, 'gui_config.json')

        # Merge rather than overwrite - setup_lastools.py writes to the same
        # file and does the same, so neither clobbers the other's keys.
        config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as handle:
                    config = json.load(handle)
            except (ValueError, OSError):
                config = {}
        if not isinstance(config, dict):
            config = {}

        config['lastools_path'] = self.lastools_path.get()
        config['appearance_mode'] = self.appearance_mode.get()
        try:
            with open(config_file, 'w') as handle:
                json.dump(config, handle, indent=2)
            return True
        except OSError:
            return False

    def save_config_interactive(self):
        """Save settings and report the outcome. Bound to the Save button."""
        # Kept separate from save_config because save_config also runs during
        # shutdown, where popping up a dialog would block the window closing.
        if self.save_config():
            messagebox.showinfo("Saved", "Settings saved.")
        else:
            messagebox.showerror("Error", "Could not write gui_config.json.")

    def open_documentation(self):
        for name in ('USAGE.md', 'README.md'):
            doc_file = os.path.join(PROJECT_ROOT, name)
            if os.path.exists(doc_file):
                try:
                    os.startfile(doc_file)
                except OSError as e:
                    messagebox.showerror("Error", f"Could not open {name}:\n{e}")
                return
        messagebox.showinfo("Not found", "No USAGE.md or README.md next to the app.")

    # -- console ------------------------------------------------------------
    def clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        self._console_transient = False

    def save_console(self):
        path = filedialog.asksaveasfilename(
            title="Save console log",
            defaultextension=".txt",
            initialfile=f"fsct_log_{datetime.now():%Y%m%d_%H%M%S}.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(self.console.get("1.0", "end"))
            self.update_status(f"Log saved to {os.path.basename(path)}")
        except OSError as e:
            messagebox.showerror("Error", f"Could not save the log:\n{e}")

    def _console_sink(self, text, transient):
        """Called by ConsoleStream from the worker thread."""
        level = "info"
        lowered = text.lower()
        if "error" in lowered or "traceback" in lowered or "failed" in lowered:
            level = "err"
        elif "done" in lowered or "complete" in lowered:
            level = "ok"

        for marker, stage in STAGE_MARKERS:
            if marker.lower() in lowered:
                self._set_stage(stage)
                break

        self._console_queue.append((text, level, transient))

    def console_log(self, message, level="info"):
        """Queue a console line. Safe to call from any thread."""
        self._console_queue.append((message, level, False))

    def log_operation(self, message):
        """Queue an operation-log line. Safe to call from any thread."""
        self._log_queue.append(message)

    def _pump_output(self):
        """
        Drain both queues onto the widgets.

        This is the only place either text widget is written, and it always
        runs on the Tk main loop. Draining in batches keeps a chatty stage
        (cylinder fitting prints per tree) from starving the event loop.
        """
        try:
            if self._pending_stage is not None:
                self.stage_label.configure(text=self._pending_stage)
                self._pending_stage = None
            # Text first, so the console is up to date before a callback puts a
            # modal dialog on screen and blocks this loop until it is dismissed.
            self._drain_console()
            self._drain_log()
            self._drain_callbacks()
        finally:
            self.root.after(120, self._pump_output)

    def _drain_callbacks(self):
        while self._callback_queue:
            func, args = self._callback_queue.popleft()
            try:
                func(*args)
            except Exception:
                import traceback
                # One broken callback must not stop the pump, or the console
                # and every later completion handler stall with it.
                self.console_log(
                    "Internal error in a UI callback:\n" + traceback.format_exc(), "err"
                )

    def _drain_console(self):
        if not self._console_queue:
            return

        self.console.configure(state="normal")
        drained = 0
        while self._console_queue and drained < 300:
            text, level, transient = self._console_queue.popleft()
            drained += 1

            if self._console_transient:
                # Replace the progress line written last time round.
                self.console.delete("end-2l linestart", "end-1c")
            self.console.insert("end", text + "\n", None if level == "info" else (level,))
            self._console_transient = transient

        # Keep the widget bounded; a long run prints far more than anyone
        # scrolls back through, and an unbounded Text widget slows Tk down.
        try:
            lines = int(self.console.index("end-1c").split(".")[0])
            if lines > 4000:
                self.console.delete("1.0", f"{lines - 3000}.0")
        except (tk.TclError, ValueError):
            pass

        self.console.see("end")
        self.console.configure(state="disabled")

    def _drain_log(self):
        if not self._log_queue:
            return

        self.operation_log.configure(state="normal")
        drained = 0
        while self._log_queue and drained < 200:
            message = self._log_queue.popleft()
            drained += 1
            self.operation_log.insert(
                "end", f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
            )
        self.operation_log.see("end")
        self.operation_log.configure(state="disabled")

    def _tick_clock(self):
        """Update the clock, and the elapsed timer while a run is going."""
        self.time_label.configure(text=datetime.now().strftime('%H:%M:%S'))
        if self.processing and self._start_time:
            elapsed = int(time.time() - self._start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_label.configure(text=f"elapsed {hours:d}:{minutes:02d}:{seconds:02d}")
        self.root.after(1000, self._tick_clock)

    # -- misc ---------------------------------------------------------------
    def _on_main_thread(self, func, *args):
        """
        Queue func(*args) to run on the Tk main loop. Safe from any thread.

        Deliberately not root.after(). Tkinter may only be touched from the
        thread that owns the interpreter: after() called from a worker relies
        on Tcl's notifier waking the main thread, and raises "main thread is
        not in main loop" outright if the loop is being driven by update()
        rather than mainloop() - which is exactly what a test harness does, so
        the failure hides until something automated tries to exercise it.
        Appending to a deque is safe from anywhere, and _pump_output - which
        genuinely runs on the main loop - is the only thing that calls back in.
        """
        self._callback_queue.append((func, args))

    def update_status(self, message):
        """Update the status bar. Safe to call from any thread."""
        self._on_main_thread(lambda: self.status_label.configure(text=message))


def claim_taskbar_identity(app_id="FSCT.ForestStructuralComplexityTool.Desktop"):
    """
    Give the app its own taskbar identity on Windows.

    Windows identifies a process for taskbar purposes by its Application User
    Model ID, and with none set it falls back to the executable - which here is
    python.exe. The taskbar button then belongs to "Python": it groups with any
    other Python window, shows the Python icon rather than this window's, and
    pinning it pins the interpreter instead of FSCT.

    Must run before the first window exists, so main() calls it ahead of CTk().
    Silently does nothing off Windows, or if the shell API is unavailable.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except (AttributeError, OSError):
        pass


def main():
    claim_taskbar_identity()
    ctk.set_default_color_theme("green")
    root = ctk.CTk()
    FSCTStandaloneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
