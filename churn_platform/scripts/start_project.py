from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_service(process: subprocess.Popen, url: str, name: str, timeout: int = 45) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} stopped before becoming ready.")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(f"{name} did not become ready within {timeout} seconds.")


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The project launcher requires the Python 3.11 virtual environment. Run start_project.bat.")

    print("Checking project dependencies...", flush=True)
    subprocess.run(
        [str(PYTHON), "-m", "pip", "install", "-q", "-r", str(PROJECT_ROOT / "requirements.txt")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("Retraining the churn model...", flush=True)
    subprocess.run([str(PYTHON), "run_training.py"], cwd=PROJECT_ROOT, check=True)

    api_port = find_free_port()
    dashboard_port = find_free_port()
    while dashboard_port == api_port:
        dashboard_port = find_free_port()

    creation_flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    processes: list[subprocess.Popen] = []
    try:
        processes.append(
            subprocess.Popen(
                [
                    str(PYTHON),
                    "-m",
                    "uvicorn",
                    "api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(api_port),
                ],
                cwd=PROJECT_ROOT,
                creationflags=creation_flags,
            )
        )
        processes.append(
            subprocess.Popen(
                [
                    str(PYTHON),
                    "-m",
                    "streamlit",
                    "run",
                    "dashboard/app.py",
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(dashboard_port),
                    "--server.headless",
                    "true",
                ],
                cwd=PROJECT_ROOT,
                creationflags=creation_flags,
            )
        )

        api_url = f"http://127.0.0.1:{api_port}"
        dashboard_url = f"http://127.0.0.1:{dashboard_port}"
        wait_for_service(processes[0], f"{api_url}/health", "Prediction API")
        wait_for_service(processes[1], f"{dashboard_url}/_stcore/health", "Dashboard")
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise

    print(f"Prediction API: {api_url}/docs")
    print(f"Dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, TimeoutError) as error:
        print(f"Startup error: {error}", file=sys.stderr)
        raise SystemExit(1)
