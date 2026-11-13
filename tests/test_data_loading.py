import tempfile
from pathlib import Path

import pandas as pd

import src.data_loading as data_loading


def test_project_root_contains_src_package():
    root = data_loading.project_root()

    assert (root / "src" / "data_loading.py").exists()
    assert data_loading.data_dir() == root / "data"


def test_loaders_read_csv_from_data_directory():
    original = data_loading.data_dir
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pd.DataFrame({"CODE": [1], "population": [10]}).to_csv(tmp_path / "demo.csv", index=False)
        pd.DataFrame({"CODE": [1], "shelter_cost": [900]}).to_csv(tmp_path / "spend.csv", index=False)
        data_loading.data_dir = lambda: tmp_path
        try:
            demo = data_loading.load_demo_data("demo.csv")
            spend = data_loading.load_spend_data("spend.csv")
        finally:
            data_loading.data_dir = original

    assert demo.to_dict("records") == [{"CODE": 1, "population": 10}]
    assert spend.to_dict("records") == [{"CODE": 1, "shelter_cost": 900}]
