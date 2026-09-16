"""
LAStools setup helper.

Downloads the official LAStools distribution, unpacks it inside the project
(third_party/LAStools) and records the path to its 'bin' directory in
gui_config.json so the GUI/wrapper can find lasview.exe, laszip.exe and
lasthin.exe without the user configuring anything by hand.

Can be used three ways:
    python setup_lastools.py            # download if missing, then report path
    python setup_lastools.py --force    # re-download even if already present
    from setup_lastools import ensure_lastools; ensure_lastools()

Note on licensing: LAStools is a mixed distribution. laszip/lasview and the
other LGPL tools are free to use; several of the other executables are
commercial and will watermark output without a licence key. FSCT only uses the
free ones. See https://rapidlasso.de/lastools/ for details.
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile

# Primary download is the official distribution zip; the GitHub release is a
# fallback in case the first host is unreachable.
LASTOOLS_URLS = [
    "https://lastools.github.io/download/LAStools.zip",
    "https://github.com/LAStools/LAStools/releases/latest/download/LAStools.zip",
]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTALL_ROOT = os.path.join(PROJECT_ROOT, "third_party")
LASTOOLS_DIR = os.path.join(INSTALL_ROOT, "LAStools")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "gui_config.json")

# Executables FSCT actually calls. lasview is the one the GUI's
# "Open in LAStools" button needs, so it is what we look for.
REQUIRED_EXE = "lasview.exe"


def find_lastools_bin(search_root=LASTOOLS_DIR):
    """Return the directory containing lasview.exe under search_root, or None."""
    if not os.path.isdir(search_root):
        return None
    for dirpath, _dirnames, filenames in os.walk(search_root):
        lowered = {f.lower() for f in filenames}
        if REQUIRED_EXE in lowered:
            return dirpath
    return None


def _ssl_context():
    """
    Build an SSL context, preferring certifi's CA bundle.

    Python's default context loads the Windows certificate store, and a single
    malformed entry there makes OpenSSL 3.x reject the whole batch with
    "[ASN1: NOT_ENOUGH_DATA]" - which breaks every HTTPS request from that
    interpreter, not just this one. certifi ships a known-good bundle and is
    already a transitive dependency of the environment, so use it when present
    and only fall back to the platform default if it is not.
    """
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    try:
        return ssl.create_default_context()
    except ssl.SSLError:
        # Platform store is unusable and certifi is unavailable. Let urlopen
        # use its own default and surface whatever error it produces.
        return None


def _download(url, dest_path, progress=None):
    """Stream url to dest_path. progress(bytes_done, bytes_total) is optional."""
    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": "FSCT-GUI-setup"})
    context = _ssl_context()
    kwargs = {"context": context} if context is not None else {}
    with urlopen(request, timeout=60, **kwargs) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(dest_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    return dest_path


def _safe_extract(archive, destination):
    """
    Extract archive to destination, refusing entries that escape it.

    A zip entry named "../../evil.exe" would otherwise write outside the
    target directory ("zip slip"). The download is from a trusted host over
    HTTPS, but the check costs nothing.
    """
    destination = os.path.abspath(destination)
    for member in archive.namelist():
        target = os.path.abspath(os.path.join(destination, member))
        if target != destination and not target.startswith(destination + os.sep):
            raise ValueError(f"Archive entry escapes the target directory: {member}")
    archive.extractall(destination)


def _default_progress(done, total):
    if total:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r  Downloading LAStools... {pct:5.1f}%  ({done / 1e6:.1f} / {total / 1e6:.1f} MB)")
    else:
        sys.stdout.write(f"\r  Downloading LAStools... {done / 1e6:.1f} MB")
    sys.stdout.flush()


def save_lastools_path(bin_dir, config_file=CONFIG_FILE):
    """Merge the LAStools bin path into gui_config.json without losing other keys."""
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as handle:
                config = json.load(handle)
        except (ValueError, OSError):
            config = {}
    if not isinstance(config, dict):
        config = {}

    config["lastools_path"] = bin_dir.replace("\\", "/")
    with open(config_file, "w") as handle:
        json.dump(config, handle, indent=2)
    return config_file


def ensure_lastools(force=False, progress=_default_progress, quiet=False):
    """
    Make sure LAStools is available locally.

    Returns the path to the LAStools 'bin' directory, or None if it could not
    be installed (which is not fatal - FSCT falls back to laspy for everything
    except the external 3D viewer).
    """

    def say(message):
        if not quiet:
            print(message)

    existing = find_lastools_bin()
    if existing and not force:
        say(f"LAStools already present: {existing}")
        save_lastools_path(existing)
        return existing

    if force and os.path.isdir(LASTOOLS_DIR):
        shutil.rmtree(LASTOOLS_DIR, ignore_errors=True)

    os.makedirs(INSTALL_ROOT, exist_ok=True)

    archive_path = os.path.join(tempfile.gettempdir(), "LAStools_fsct_download.zip")
    last_error = None
    for url in LASTOOLS_URLS:
        try:
            say(f"Fetching LAStools from {url}")
            _download(url, archive_path, progress=None if quiet else progress)
            if not quiet:
                print()
            break
        except Exception as error:  # network error, 404, timeout, ...
            last_error = error
            say(f"  Failed: {error}")
    else:
        say("Could not download LAStools. FSCT will still work; the external")
        say("3D viewer button will be unavailable until you install it manually")
        say("from https://rapidlasso.de/lastools/")
        if last_error:
            say(f"Last error: {last_error}")
        return None

    say("Extracting...")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, LASTOOLS_DIR)
    except (zipfile.BadZipFile, ValueError) as error:
        say(f"Could not extract the downloaded archive: {error}")
        return None
    finally:
        try:
            os.remove(archive_path)
        except OSError:
            pass

    # The distribution zip has no single top-level folder, so it is extracted
    # into LASTOOLS_DIR rather than INSTALL_ROOT - otherwise its bin/, data/,
    # *_toolbox/ and assorted .txt files land loose in third_party/.
    bin_dir = find_lastools_bin()
    if not bin_dir:
        say(f"Extracted LAStools but could not find {REQUIRED_EXE} inside {LASTOOLS_DIR}.")
        return None

    save_lastools_path(bin_dir)
    say(f"LAStools ready: {bin_dir}")
    say(f"Path saved to {CONFIG_FILE}")
    return bin_dir


def main():
    force = "--force" in sys.argv
    bin_dir = ensure_lastools(force=force)
    return 0 if bin_dir else 1


if __name__ == "__main__":
    sys.exit(main())
