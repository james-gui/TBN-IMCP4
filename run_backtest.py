#!/usr/bin/env python3
"""
Run backtest and automatically open results in the local visualizer.

Usage:
    python run_backtest.py [day] [--algo PATH] [--data DIR] [--merge-pnl]

Examples:
    python run_backtest.py 0
    python run_backtest.py 0 --merge-pnl
    python run_backtest.py 0 --algo viz/attempt2_viz.py
    python run_backtest.py 0 --data data/          # explicit data dir
    python run_backtest.py          # defaults to day 0

If a data/ directory exists at the project root it is used automatically
(equivalent to passing --data data/).

First-time setup (build the visualizer once):
    python setup_visualizer.py
"""

import http.server
import os
import subprocess
import sys
import threading
import time
import webbrowser

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_ALGORITHM = "viz/message_viz.py"
OUTPUT_FILE  = "output.log"
SERVER_PORT  = 8765

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DIST_DIR       = os.path.join(BASE_DIR, "visualizer", "dist")
# The visualizer is built with base="/imc-prosperity-3-visualizer/" by Vite
VISUALIZER_URL_PATH = "/imc-prosperity-3-visualizer/"
# ──────────────────────────────────────────────────────────────────────────────


class LocalHandler(http.server.BaseHTTPRequestHandler):
    """
    Single-server routing:
      /output.log                        → BASE_DIR/output.log
      /imc-prosperity-3-visualizer/...   → DIST_DIR/...
    Both served with CORS headers so the visualizer can fetch the log file.
    """

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query string for file lookup

        if path == "/output.log":
            self._serve_file(os.path.join(BASE_DIR, OUTPUT_FILE))

        elif path.startswith(VISUALIZER_URL_PATH):
            # Strip the base prefix, default to index.html
            rel = path[len(VISUALIZER_URL_PATH):]
            if not rel:
                rel = "index.html"
            self._serve_file(os.path.join(DIST_DIR, rel))

        elif path in ("/", ""):
            # Redirect bare root to the visualizer
            self.send_response(302)
            self.send_header("Location", VISUALIZER_URL_PATH)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self._add_cors()
        self.end_headers()

    def _serve_file(self, filepath: str):
        if not os.path.isfile(filepath):
            self.send_response(404)
            self.end_headers()
            return

        with open(filepath, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", self._guess_type(filepath))
        self._add_cors()
        self.end_headers()
        self.wfile.write(data)

    def _add_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    @staticmethod
    def _guess_type(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".html": "text/html",
            ".js":   "application/javascript",
            ".css":  "text/css",
            ".json": "application/json",
            ".svg":  "image/svg+xml",
            ".png":  "image/png",
            ".ico":  "image/x-icon",
            ".log":  "text/plain",
        }.get(ext, "application/octet-stream")

    def log_message(self, format, *args):
        pass  # suppress per-request noise


def check_visualizer_built():
    if not os.path.isdir(DIST_DIR):
        print("Visualizer not built yet. Run first:")
        print("    python setup_visualizer.py")
        sys.exit(1)


def run_server(port: int) -> http.server.HTTPServer:
    server = http.server.HTTPServer(("localhost", port), LocalHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    check_visualizer_built()

    args = sys.argv[1:]
    day = "0"
    algo = DEFAULT_ALGORITHM
    data_dir = None
    extra_flags = []
    i = 0
    while i < len(args):
        if args[i] == "--algo" and i + 1 < len(args):
            algo = args[i + 1]
            i += 2
        elif args[i] == "--data" and i + 1 < len(args):
            data_dir = args[i + 1]
            i += 2
        elif args[i].startswith("--"):
            extra_flags.append(args[i])
            i += 1
        else:
            day = args[i]
            i += 1

    # Auto-use local data/ directory if present and --data wasn't given
    if data_dir is None:
        default_data = os.path.join(BASE_DIR, "data")
        if os.path.isdir(default_data):
            data_dir = default_data

    data_flags = ["--data", data_dir] if data_dir else []
    cmd = ["prosperity4btx", algo, day, "--out", OUTPUT_FILE] + data_flags + extra_flags
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, shell=(sys.platform == "win32"))
    if result.returncode != 0:
        print("Backtest failed — check the output above.")
        sys.exit(result.returncode)

    output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
    if not os.path.exists(output_path):
        print(f"Expected output file not found: {output_path}")
        sys.exit(1)

    print(f"Backtest complete. Starting local server on port {SERVER_PORT}...")
    run_server(SERVER_PORT)
    time.sleep(0.3)

    log_url        = f"http://localhost:{SERVER_PORT}/output.log"
    visualizer_url = f"http://localhost:{SERVER_PORT}{VISUALIZER_URL_PATH}?open={log_url}"

    print(f"Opening: {visualizer_url}")
    webbrowser.open(visualizer_url)

    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
