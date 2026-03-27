import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_logistic_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a logistic regression baseline and return metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds, output_dict=True),
    }
