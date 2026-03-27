import pandas as pd


def merge_on_code(demo_df: pd.DataFrame, spend_df: pd.DataFrame, key: str = "CODE") -> pd.DataFrame:
    """Merge two dataframes on shared geographic code."""
    return demo_df.merge(spend_df, on=key, how="inner")


def build_target_column(
    df: pd.DataFrame,
    rent_col: str = "shelter_cost",
    income_col: str = "household_income",
    threshold: float = 0.3,
    target_col: str = "y",
) -> pd.DataFrame:
    """
    Create a binary housing-burden target where 1 means burdened.

    y = 1 if shelter_cost / household_income >= threshold else 0
    """
    out = df.copy()
    ratio = out[rent_col] / out[income_col]
    out[target_col] = (ratio >= threshold).astype(int)
    return out


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns if they exist (safe for evolving schemas)."""
    return df.drop(columns=[c for c in columns if c in df.columns], errors="ignore")
