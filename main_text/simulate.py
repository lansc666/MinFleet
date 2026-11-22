"""Run main-text baseline, gravity and vehicle-waiting experiments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.simulation_runner import run

if __name__ == "__main__":
    run("main_text", ["baseline", "alpha", "beta", "gamma", "vehicle_wait"])
