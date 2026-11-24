"""Training and evaluation: nested cross-validation and hold-out evaluation."""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from . import config
from .models import build_pipeline, model_specs, pca_label

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "auc": "roc_auc",
}


def _grid_search(pipeline, grid):
    """GridSearchCV with the inner stratified folds from the config."""
    inner = StratifiedKFold(n_splits=config.INNER_FOLDS, shuffle=True,
                            random_state=config.RANDOM_STATE)
    return GridSearchCV(pipeline, grid, cv=inner,
                        scoring=config.SEARCH_SCORING, n_jobs=-1)


def nested_cv(X, y):
    """Nested CV for every (model, PCA setting) pair.

    The outer loop estimates generalisation performance; the inner loop
    (GridSearchCV) picks hyperparameters using only the outer training folds.

    Returns a DataFrame with mean and std of each metric across outer folds.
    """
    outer = StratifiedKFold(n_splits=config.OUTER_FOLDS, shuffle=True,
                            random_state=config.RANDOM_STATE)
    rows = []
    for name in model_specs():
        for n_comp in config.PCA_SETTINGS:
            pipe, grid = build_pipeline(name, n_comp)
            search = _grid_search(pipe, grid)
            scores = cross_validate(search, X, y, cv=outer, scoring=SCORING)
            row = {"model": name, "features": pca_label(n_comp), "n_components": n_comp}
            for metric in SCORING:
                values = scores[f"test_{metric}"]
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
            rows.append(row)
            print(f"[nested CV] {name:14s} {pca_label(n_comp):16s} "
                  f"F1={row['f1_mean']:.3f}  recall={row['recall_mean']:.3f}")
    return pd.DataFrame(rows)


def positive_scores(model, X):
    """Continuous score for the positive (malignant) class, used for ROC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def holdout(X_train, X_test, y_train, y_test, n_components=None):
    """Tune each model on the training split and evaluate on the test split.

    Returns
    -------
    fitted : dict name -> fitted GridSearchCV
    metrics : DataFrame with one row per model
    details : dict name -> {"y_pred", "y_score", "confusion"}
    """
    fitted, rows, details = {}, [], {}
    for name in model_specs():
        pipe, grid = build_pipeline(name, n_components)
        search = _grid_search(pipe, grid).fit(X_train, y_train)
        y_pred = search.predict(X_test)
        y_score = positive_scores(search, X_test)

        fitted[name] = search
        details[name] = {
            "y_pred": y_pred,
            "y_score": y_score,
            "confusion": confusion_matrix(y_test, y_pred),
        }
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_score),
            "best_params": str(search.best_params_),
        })
    return fitted, pd.DataFrame(rows), details


def best_configuration(cv_results):
    """Pick the (model, features) row with the highest mean F1 in nested CV.

    Ties are broken by recall, which matters most for malignant tumours.
    """
    ranked = cv_results.sort_values(["f1_mean", "recall_mean"], ascending=False)
    return ranked.iloc[0]


def misclassified(X_test, y_test, y_pred):
    """Test samples the model got wrong (for the qualitative analysis)."""
    wrong = np.where(y_test != y_pred)[0]
    out = X_test.iloc[wrong].copy()
    out["true"] = np.where(y_test[wrong] == 1, "malignant", "benign")
    out["predicted"] = np.where(y_pred[wrong] == 1, "malignant", "benign")
    return out
