"""
run_pipeline.py
Runs the full churn prediction pipeline in sequence.
Execute from the project root — works regardless of where scripts live.
"""

import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

SCRIPTS = [
    "data_cleaning.py",
    "feature_engineering.py",
    "train.py",
    "evaluate.py",
    "visualize.py",
    "predict.py",
]


def run_script(script: str):
    """Run a single script and stop the pipeline if it fails."""
    print(f"\n{'='*50}")
    print(f"Running: {script}")
    print(f"{'='*50}")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        cwd=SCRIPTS_DIR
    )

    if result.returncode != 0:
        print(f"\nPipeline stopped — {script} failed.")
        sys.exit(result.returncode)

    print(f"Done: {script}")


if __name__ == "__main__":
    print("Starting Telco Churn Prediction Pipeline...")
    print(f"Scripts folder : {SCRIPTS_DIR}")

    for script in SCRIPTS:
        run_script(script)

    print("\nPipeline complete.")
    print("Model saved to          : models/")
    print("Charts saved to         : outputs/visualizations/")
    print("Predictions ready via   : scripts/predict.py")