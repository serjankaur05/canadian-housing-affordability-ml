from pathlib import Path

import pandas as pd


def project_root() -> Path:
    """Return repository root based on this file location."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Return local data directory path."""
    return project_root() / "data"


def load_demo_data(filename: str = "DemoStats_(DST)_2020_2025_GEO.csv") -> pd.DataFrame:
    """Load demographic dataset from local data directory."""
    path = data_dir() / filename
    return pd.read_csv(path)


def load_spend_data(filename: str = "HouseholdSpend_2024_GEO.csv") -> pd.DataFrame:
    """Load household spending dataset from local data directory."""
    path = data_dir() / filename
    return pd.read_csv(path)
