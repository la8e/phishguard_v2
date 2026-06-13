#!/usr/bin/env python3
"""
PhishGuard Setup Script
Cross-platform installer for Windows, Linux, and macOS.

Handles:
- Git availability check
- Repository cloning
- Virtual environment creation
- Dependency installation with XGBoost CPU-only
- Disk space check
- FastText model download with progress and resume support
- Partial download cleanup on failure
"""

import gzip
import shutil
import subprocess
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


# CONFIG
REPO_URL     = "https://github.com/la8e/phishguard_v2.git"
REPO_NAME    = "phishguard_v2"
FASTTEXT_URL = ("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz")
FASTTEXT_GZ_SIZE_APPROX = 4_200_000_000   # ~4.2 GB compressed
FASTTEXT_BIN_SIZE_APPROX = 7_000_000_000  # ~7.0 GB decompressed
MIN_FREE_DISK = FASTTEXT_GZ_SIZE_APPROX + FASTTEXT_BIN_SIZE_APPROX + 500_000_000
XGBOOST_VERSION = "xgboost==3.2.0"
MAX_VT_RETRIES  = 5   # not used here but documents the constant for vt_client


# UTILITIES
def run(cmd: list, cwd: Path = None) -> None:
    """Run a subprocess command; exit with an error message on failure."""
    try:
        subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {' '.join(str(c) for c in cmd)}")
        print(f"  Exit code: {e.returncode}")
        sys.exit(1)

def check_git() -> None:
    """Verify Git is installed and accessible on PATH."""
    try:
        subprocess.check_call(
            ["git", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[ERROR] Git is not installed or not on PATH.")
        print("  Download it from https://git-scm.com/downloads")
        sys.exit(1)


def check_disk_space(target_dir: Path) -> None:
    """
    Verify there is enough free disk space to download and extract
    the FastText model before starting the download.
    Requires ~11.5 GB free: 4.2 GB compressed + 7.0 GB extracted + buffer.
    """
    free = shutil.disk_usage(target_dir).free
    free_gb = free / 1024 ** 3
    needed_gb = MIN_FREE_DISK / 1024 ** 3
    if free < MIN_FREE_DISK:
        print(f"[ERROR] Insufficient disk space.")
        print(f"  Available : {free_gb:.1f} GB")
        print(f"  Required  : {needed_gb:.1f} GB")
        sys.exit(1)
    print(f"Disk space OK: {free_gb:.1f} GB available.")


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Print a live download progress line using carriage return."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        downloaded_mb = downloaded / 1_000_000
        total_mb = total_size / 1_000_000
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  "
            f"{downloaded_mb:,.0f} / {total_mb:,.0f} MB"
        )
    else:
        downloaded_mb = downloaded / 1_000_000
        sys.stdout.write(f"\r  Downloaded: {downloaded_mb:,.0f} MB")
    sys.stdout.flush()


def get_python_exe(venv_dir: Path) -> Path:
    """Return the path to the Python executable inside the virtual environment."""
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def get_activation_hint(venv_dir: Path) -> str:
    """Return the platform-appropriate venv activation command."""
    if sys.platform.startswith("win"):
        return str(venv_dir / "Scripts" / "activate")
    return f"source {venv_dir / 'bin' / 'activate'}"


# STEP 1 - GIT CHECK
print("PhishGuard Installer")
print()
print("[1/6] Checking Git installation...")
check_git()
print("Git found.")


# STEP 2 - CLONE REPOSITORY
print("[2/6] Cloning repository...")
repo_dir = Path(REPO_NAME)
if not repo_dir.exists():
    run(["git", "clone", REPO_URL])
    print("Repository cloned.")
else:
    print("Repository already exists. Skipping clone.")


# STEP 3 - VIRTUAL ENVIRONMENT
print("[3/6] Creating virtual environment...")
venv_dir = repo_dir / ".venv"
if not venv_dir.exists():
    run([sys.executable, "-m", "venv", str(venv_dir)])
    print("Virtual environment created.")
else:
    print("Virtual environment already exists. Skipping.")

python_exe = get_python_exe(venv_dir)
if not python_exe.exists():
    print(f"[ERROR] Python executable not found at {python_exe}")
    sys.exit(1)


# STEP 4 - INSTALL DEPENDENCIES
print("[4/6] Installing dependencies...")

req_path = repo_dir / "requirements.txt"
if not req_path.exists():
    print(f"[ERROR] requirements file not found at {req_path}")
    sys.exit(1)

# Upgrade pip first
run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])

# Install all requirements EXCEPT xgboost (handled separately below)
run([
    str(python_exe), "-m", "pip", "install",
    "-r", str(req_path),
])

# Install XGBoost without CUDA transitive dependencies.
# --no-deps prevents pip from pulling the full CUDA stack onto a CPU machine.
print("Installing XGBoost (CPU-only)...")
run([
    str(python_exe), "-m", "pip", "install",
    XGBOOST_VERSION,
    "--no-deps",
    "--force-reinstall",
])
# Ensure XGBoost's actual runtime requirements are present
run([
    str(python_exe), "-m", "pip", "install",
    "numpy>=1.22", "scipy>=1.0",
    "--quiet",
])
print("Dependencies installed.")


# STEP 5 - FASTTEXT DIRECTORY
print("[5/6] Preparing FastText model directory...")
fasttext_dir = repo_dir / "src" / "features" / "fastText"
fasttext_dir.mkdir(parents=True, exist_ok=True)
model_path  = fasttext_dir / "cc.en.300.bin"
gz_path     = fasttext_dir / "cc.en.300.bin.gz"


# STEP 6 - DOWNLOAD + EXTRACT FASTTEXT
if model_path.exists():
    print("[6/6] FastText model already exists. Skipping download.")
else:
    print("[6/6] Downloading FastText model (~4.2 GB). Do not interrupt.")
    # Pre-flight: disk space check before committing to the download
    check_disk_space(fasttext_dir)
    print()

    # Resume support: if a partial .gz exists, report its size
    if gz_path.exists():
        existing_mb = gz_path.stat().st_size / 1_000_000
        print(f"Partial download found ({existing_mb:,.0f} MB). Restarting...")
        gz_path.unlink()

    # Download with cleanup on any failure
    try:
        urlretrieve(FASTTEXT_URL, str(gz_path), reporthook=download_progress)
        print()
    except (URLError, OSError, KeyboardInterrupt) as e:
        print(f"\n[ERROR] Download failed: {e}")
        if gz_path.exists():
            gz_path.unlink()
            print("Partial download cleaned up.")
        sys.exit(1)

    # Verify the downloaded file is not empty or truncated
    if not gz_path.exists() or gz_path.stat().st_size < 1_000_000:
        print("[ERROR] Downloaded file is missing or too small. Aborting.")
        if gz_path.exists():
            gz_path.unlink()
        sys.exit(1)

    # Extract - streams directly to destination, no full in-memory load
    print()
    print("Extracting model (this may take several minutes)...")
    try:
        with gzip.open(str(gz_path), "rb") as fin:
            with open(str(model_path), "wb") as fout:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)  # 16 MB chunks
    except (OSError, EOFError, KeyboardInterrupt) as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        # Clean up both the partial .bin and the .gz
        if model_path.exists():
            model_path.unlink()
            print("Partial model file cleaned up.")
        if gz_path.exists():
            gz_path.unlink()
        sys.exit(1)

    # Remove the compressed archive once extraction is confirmed
    if gz_path.exists():
        gz_path.unlink()
    print("FastText model extracted successfully.")

# DONE
print("Setup completed successfully.")
print("- To run PhishGuard:")
print(f" {get_activation_hint(venv_dir)}")
print(f" python src/predictor.py")
print("- With VirusTotal enrichment:")
print(f" python src/predictor.py --vt-key YOUR_API_KEY")
