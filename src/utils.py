"""Shared project paths and plotting utilities."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Cantor_machineclean-opendepository.xlsx"
FIGURES_PATH = ROOT / "outputs" / "figures"
MODELS_PATH = ROOT / "outputs" / "models"
RESULTS_PATH = ROOT / "outputs" / "results"


def ensure_output_dirs() -> None:
    for path in (FIGURES_PATH, MODELS_PATH, RESULTS_PATH):
        path.mkdir(parents=True, exist_ok=True)
