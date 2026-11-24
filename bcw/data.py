"""Dataset loading and basic description."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from . import config


def load_data():
    """Load Breast Cancer Wisconsin (Diagnostic) from scikit-learn.

    scikit-learn encodes malignant = 0 and benign = 1. We flip the labels so
    that malignant = 1 (positive class). Then precision/recall/F1 refer to
    detecting malignant tumours, which is the clinically relevant class.

    Returns
    -------
    X : pd.DataFrame, shape (569, 30)
    y : np.ndarray of {0 (benign), 1 (malignant)}
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    y = (1 - raw.target).to_numpy()
    return X, y


def describe(X, y):
    """Return a small dict with dataset facts for the article."""
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_malignant": int(np.sum(y == 1)),
        "n_benign": int(np.sum(y == 0)),
        "missing_values": int(X.isna().sum().sum()),
    }


def split(X, y):
    """Stratified hold-out split (keeps the class proportions)."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def as_frame(X, y):
    """Features plus a readable label column (used for EDA plots)."""
    df = X.copy()
    df["diagnosis"] = pd.Series(y, index=X.index).map({0: "benign", 1: "malignant"})
    return df
