import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    """Set shared plotting style for notebook outputs."""
    sns.set_theme(style="whitegrid", context="talk")


def plot_target_balance(df: pd.DataFrame, target_col: str = "y") -> None:
    """Simple count plot for binary target distribution."""
    set_style()
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target_col)
    plt.title("Housing Burden Target Distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
