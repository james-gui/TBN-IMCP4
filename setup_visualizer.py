#!/usr/bin/env python3
"""
One-time setup: install and build the local visualizer.

Requirements:
    - Node.js 18+  (https://nodejs.org)
    - pnpm         (npm install -g pnpm)

Run once after cloning:
    python setup_visualizer.py
"""

import os
import shutil
import subprocess
import sys

VISUALIZER_DIR = os.path.join(os.path.dirname(__file__), "visualizer")
DIST_DIR = os.path.join(VISUALIZER_DIR, "dist")


def check(cmd: str, install_hint: str) -> None:
    if not shutil.which(cmd):
        print(f"ERROR: '{cmd}' not found. {install_hint}")
        sys.exit(1)


def run(args: list[str], cwd: str) -> None:
    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(args)}")
        sys.exit(result.returncode)


def main():
    # Make sure submodule is initialised
    if not os.path.exists(os.path.join(VISUALIZER_DIR, "package.json")):
        print("Visualizer submodule not found. Initialising...")
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=os.path.dirname(__file__))

    check("node", "Install Node.js 18+ from https://nodejs.org")
    check("pnpm", "Run:  npm install -g pnpm")

    print("Installing visualizer dependencies...")
    run(["pnpm", "install"], cwd=VISUALIZER_DIR)

    print("Building visualizer...")
    run(["pnpm", "build"], cwd=VISUALIZER_DIR)

    if os.path.isdir(DIST_DIR):
        print(f"\nDone. Visualizer built at: {DIST_DIR}")
        print("You can now run:  python run_backtest.py 0")
    else:
        print("Build completed but dist/ not found — check for errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
