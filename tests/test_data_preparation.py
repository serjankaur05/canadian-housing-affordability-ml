import pandas as pd

from src.data_preparation import build_target_column, drop_columns, merge_on_code


def test_merge_on_code_keeps_only_shared_areas():
    demo = pd.DataFrame({"CODE": [1, 2, 3], "population": [100, 200, 300]})
    spend = pd.DataFrame({"CODE": [2, 3, 4], "shelter_cost": [900, 1200, 800]})

    merged = merge_on_code(demo, spend)

    assert sorted(merged["CODE"]) == [2, 3]
    assert {"population", "shelter_cost"} <= set(merged.columns)


def test_build_target_marks_burden_at_threshold():
    df = pd.DataFrame(
        {
            "shelter_cost": [200, 300, 400],
            "household_income": [1000, 1000, 1000],
        }
    )

    out = build_target_column(df)

    # Ratios are 0.2, 0.3 and 0.4: the 0.30 line itself counts as burdened.
    assert out["y"].tolist() == [0, 1, 1]


def test_build_target_respects_custom_threshold_and_names():
    df = pd.DataFrame({"rent": [500, 500], "income": [1000, 2000]})

    out = build_target_column(
        df, rent_col="rent", income_col="income", threshold=0.5, target_col="burdened"
    )

    assert out["burdened"].tolist() == [1, 0]


def test_build_target_does_not_modify_input():
    df = pd.DataFrame({"shelter_cost": [300], "household_income": [1000]})

    build_target_column(df)

    assert "y" not in df.columns


def test_drop_columns_ignores_missing_names():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    out = drop_columns(df, ["b", "not_a_column"])

    assert list(out.columns) == ["a", "c"]
    assert list(df.columns) == ["a", "b", "c"]
