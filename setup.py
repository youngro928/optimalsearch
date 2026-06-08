"""
setup.py — Environment Setup Script

Creates a dedicated virtual environment for the Optimal Search Path Planning
framework, upgrades pip, and installs all required dependencies inside it.

Usage:
    python setup.py

After setup, activate the environment and run the simulation:

    Windows:
        .venv/Scripts/activate
        python main.py

    macOS / Linux:
        source .venv/bin/activate
        python main.py

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import sys
import subprocess
import shutil
import venv
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PYTHON_MIN = (3, 9)

VENV_DIR = Path(".venv")

REQUIRED_PACKAGES = [
    ("numpy",      "numpy>=1.24"),
    ("matplotlib", "matplotlib>=3.7"),
    # ("scipy",      "scipy>=1.10"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_python_version():
    current = sys.version_info[:2]
    if current < PYTHON_MIN:
        print(f"  ERROR: Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]} or later is required.")
        print(f"         Detected: Python {current[0]}.{current[1]}")
        sys.exit(1)
    print(f"  [OK] Python {current[0]}.{current[1]}.{sys.version_info[2]}")


def _pip_is_functional(venv_python: Path) -> bool:
    """Return True if pip responds correctly inside the given venv."""
    result = subprocess.run(
        [venv_python, "-m", "pip", "--version"],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _bootstrap_pip(venv_python: Path) -> bool:
    """
    Try to bootstrap pip via ensurepip.
    Returns True on success, False otherwise.
    """
    result = subprocess.run(
        [venv_python, "-m", "ensurepip", "--upgrade"],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def create_virtual_environment():
    """
    Create the virtual environment, rebuilding it automatically if the
    existing one has a broken pip (e.g. after a Python version upgrade).
    """
    venv_python = get_venv_python()

    if VENV_DIR.exists():
        if _pip_is_functional(venv_python):
            print(f"  [OK] Virtual environment already exists at '{VENV_DIR}/'")
            print(f"       pip is functional — skipping creation.")
            return
        else:
            print(f"  WARNING: Existing venv at '{VENV_DIR}/' has a broken pip.")
            print(f"           Removing and rebuilding...")
            shutil.rmtree(VENV_DIR)

    print(f"  Creating virtual environment at '{VENV_DIR}/'...")
    venv.create(VENV_DIR, with_pip=True)
    print(f"  [OK] Virtual environment created.")

    # Explicitly bootstrap pip as a safety net (needed on some macOS/Homebrew setups)
    if not _pip_is_functional(venv_python):
        print("  pip not found after venv creation — running ensurepip...")
        if _bootstrap_pip(venv_python):
            print("  [OK] pip bootstrapped via ensurepip.")
        else:
            print("  ERROR: ensurepip also failed. Cannot install packages.")
            print("         Try: python -m ensurepip --upgrade")
            sys.exit(1)


def get_venv_python():
    """Return the path to the Python executable inside the virtual environment."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"


def upgrade_pip(venv_python):
    print("\n  Upgrading pip...\n")
    result = subprocess.run(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        check=False
    )

    if result.returncode != 0:
        print("  WARNING: pip upgrade failed. Continuing with current version.")
    else:
        try:
            out = subprocess.check_output(
                [venv_python, "-m", "pip", "--version"], text=True
            )
            pip_version = out.split()[1]
            print(f"  [OK] pip {pip_version}")
        except Exception:
            print("  [OK] pip upgraded")


def install_packages(venv_python):
    print("\n  Installing required packages...\n")
    pip_specs = [spec for _, spec in REQUIRED_PACKAGES]

    result = subprocess.run(
        [venv_python, "-m", "pip", "install", "--upgrade"] + pip_specs,
        check=False
    )

    if result.returncode != 0:
        print("\n  ERROR: Installation failed. Try manually inside the environment:")
        print(f"    pip install {' '.join(pip_specs)}")
        sys.exit(1)


def verify_imports(venv_python):
    print("\n  Verifying installations...\n")
    all_ok = True

    for package_name, _ in REQUIRED_PACKAGES:
        check_script = (
            f"import {package_name}, sys; "
            f"print(f'  [OK] {package_name} ' + {package_name}.__version__)"
        )
        result = subprocess.run(
            [venv_python, "-c", check_script],
            check=False
        )
        if result.returncode != 0:
            print(f"  [FAIL] {package_name} could not be imported")
            all_ok = False

    return all_ok


def print_activation_instructions():
    """Print platform-appropriate activation instructions."""
    if sys.platform == "win32":
        activate_cmd = r".venv\Scripts\activate"
    else:
        activate_cmd = "source .venv/bin/activate"

    print("=" * 55)
    print(" Setup complete.")
    print()
    print(" Activate the environment and run the simulation:")
    print()
    print(f"   {activate_cmd}")
    print(f"   python main.py")
    print()
    print(" To deactivate the environment when finished:")
    print()
    print("   deactivate")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print(" Optimal Search Path Planning — Environment Setup")
    print("=" * 55)
    print()

    # Step 1: Check Python version
    print("[1/4] Checking Python version...")
    check_python_version()

    # Step 2: Create virtual environment (rebuilds if pip is broken)
    print("\n[2/4] Setting up virtual environment...")
    create_virtual_environment()
    venv_python = get_venv_python()

    if not venv_python.exists():
        print(f"  ERROR: Virtual environment Python not found at '{venv_python}'")
        sys.exit(1)

    # Step 3: Upgrade pip inside the venv
    print("\n[3/4] Upgrading pip...")
    upgrade_pip(venv_python)

    # Step 4: Install and verify packages
    print("\n[4/4] Installing dependencies...")
    install_packages(venv_python)
    ok = verify_imports(venv_python)

    print()
    if ok:
        print_activation_instructions()
    else:
        print("  Setup finished with errors. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()