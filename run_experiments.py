"""Run every experiment and save figures + tables.

Usage:
    python run_experiments.py            # full run (nested CV, a few minutes)
    python run_experiments.py --quick    # fewer folds, for a fast smoke test
"""
import argparse
import json

from bcw import config, data, evaluate, plots
from bcw.models import model_specs


def main(quick=False):
    if quick:
        config.OUTER_FOLDS, config.INNER_FOLDS = 3, 2
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data and exploratory analysis ------------------------------------
    X, y = data.load_data()
    info = data.describe(X, y)
    print("Dataset:", info)
    plots.class_balance(y)
    plots.correlation_heatmap(X)
    info["pca_components_for_95pct_variance"] = plots.pca_variance(X)
    plots.pca_scatter(X, y)

    # 2. Nested CV: the main quantitative result --------------------------
    cv_results = evaluate.nested_cv(X, y)
    cv_results.to_csv(config.RESULTS_DIR / "nested_cv_results.csv", index=False)
    plots.metric_bars(cv_results, "f1", "05_f1_by_model_and_pca",
                      "F1-score for malignant tumours")
    plots.metric_bars(cv_results, "recall", "06_recall_by_model_and_pca",
                      "Recall for malignant tumours (missed tumours are the costly error)")

    # 3. Hold-out evaluation for ROC, confusion matrix and error analysis --
    X_train, X_test, y_train, y_test = data.split(X, y)
    _, holdout_metrics, details = evaluate.holdout(X_train, X_test, y_train, y_test)
    holdout_metrics.to_csv(config.RESULTS_DIR / "holdout_metrics.csv", index=False)
    plots.roc_curves(y_test, details)

    best = evaluate.best_configuration(cv_results)
    info["best_by_nested_cv"] = {"model": best["model"], "features": best["features"],
                                 "f1": round(best["f1_mean"], 4),
                                 "recall": round(best["recall_mean"], 4)}
    print("\nBest configuration:", info["best_by_nested_cv"])

    # Confusion matrix and errors of the best model, with its own PCA setting
    n_comp = None if best["n_components"] != best["n_components"] else int(best["n_components"])
    fitted_best, _, details_best = evaluate.holdout(
        X_train, X_test, y_train, y_test, n_components=n_comp)
    plots.confusion_matrix_plot(details_best[best["model"]]["confusion"],
                                f"{best['model']} / {best['features']}")
    wrong = evaluate.misclassified(X_test, y_test, details_best[best["model"]]["y_pred"])
    wrong.to_csv(config.RESULTS_DIR / "misclassified_samples.csv")
    print(f"{len(wrong)} test samples misclassified by the best model")

    # 4. Decision regions in the PCA-2D plane ------------------------------
    fitted_2d, _, _ = evaluate.holdout(X_train, X_test, y_train, y_test, n_components=2)
    plots.decision_boundaries(fitted_2d, X_test, y_test)

    # 5. Save the summary used to write the article ------------------------
    info["best_params_holdout"] = {n: str(f.best_params_) for n, f in fitted_best.items()}
    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("\nDone. Figures in ./figures, tables in ./results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke test")
    main(parser.parse_args().quick)
