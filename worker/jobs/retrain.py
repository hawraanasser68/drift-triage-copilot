"""
Retrain job: re-run train.py and register a new model version.
The new version lands in Staging — promotion to Production still
requires a human via the normal gate.
"""

import os
import subprocess
import sys


def run(payload: dict):
    """
    payload keys (all optional):
      - investigation_id
      - triggered_by
    """
    print(f"Starting retrain (triggered by investigation {payload.get('investigation_id')}) ...")

    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True,
        text=True,
        cwd=os.getenv("APP_DIR", "."),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"train.py exited with code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    print("Retrain complete.")
    print(result.stdout)
