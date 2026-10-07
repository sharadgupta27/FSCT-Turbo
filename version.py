"""
Single source of truth for the FSCT GUI version.

Every surface that shows a version - the desktop sidebar,
`FSCT-Turbo.bat --version`, `batch_process.py --version` and the installation test -
reads it from here, so a release is a one-line change. FSCT-Turbo.bat parses this file
with findstr rather than importing it, since it needs the version before the
conda environment necessarily exists; keep the assignment on one line and in the
`__version__ = "X.Y.Z"` form so that parse keeps working.

Versioning is semantic: MAJOR for changes that alter measurements or break an
existing workflow, MINOR for new capability that leaves results alone, PATCH for
fixes with no effect on output.
"""

__version__ = "1.0.0"
