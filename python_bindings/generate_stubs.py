#!/usr/bin/env python3
"""
Generate type stubs for Factorama Python bindings

This script generates .pyi files that VSCode and other IDEs can use for
intellisense and type checking with the compiled pybind11 extension.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to the python bindings directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Ensure pybind11-stubgen is available
    try:
        import pybind11_stubgen
    except ImportError:
        print("Installing pybind11-stubgen...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11-stubgen>=0.10.0"])

    # Generate stubs for the factorama module
    print("Generating type stubs for factorama...")

    # Run pybind11-stubgen
    cmd = [
        sys.executable, "-m", "pybind11_stubgen",
        "factorama",
        "-o", "python",
        "--root-suffix", "",
        "--ignore-invalid-expressions", ".*"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Type stubs generated successfully!")
        print(f"Stubs location: {script_dir}/python/factorama/")

        # List generated files
        stub_dir = script_dir / "python" / "factorama"
        if stub_dir.exists():
            pyi_files = list(stub_dir.glob("*.pyi"))
            if pyi_files:
                print("Generated stub files:")
                for f in pyi_files:
                    print(f"  - {f.name}")
            else:
                print("No .pyi files found - checking subdirectories...")
                for pyi_file in stub_dir.rglob("*.pyi"):
                    print(f"  - {pyi_file.relative_to(stub_dir)}")

    except subprocess.CalledProcessError as e:
        print(f"Error generating stubs: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command stderr: {e.stderr}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())