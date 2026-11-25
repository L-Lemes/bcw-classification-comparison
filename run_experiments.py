"""Run every experiment and save figures + tables.

Protocol (the hold-out set is never used to choose anything):
  1. Split the data once into train (75%) and test (25%), stratified.
  2. Nested cross-validation on the TRAIN split only, for all 24 configurations
     (6 models x 4 input settings). Hyperparameters are tuned in the inner loop.
  3. Select the best configuration from the nested-CV results (train only).
  4. Evaluate on the untouched hold-out set: first for the selected
     configuration, plus ROC curves and decision regions for illustration.

Usage:
    python run_experiments.py            # full run (a few minutes)
    python run_experiments.py --quick    # fewer folds, for a fast smoke test
"""
import argparse
import json

from bcw import config, data, evaluate, plots


def main(quick=False):
    if quick:
        config.OUTER_FOLDS, config.INNER_FOLDS = 3, 2
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data, single train/test split and descriptive plots ---------------
    X, y = data.load_data()
    info = data.describe(X, y)
    X_train, X_test, y_train, y_test = data.split(X, y)
    info["n_train"], info["n_test"] = int(len(y_train)), int(len(y_test))
    info["n_test_malignant"] = int(y_test.sum())
    print("Dataset:", info)

    # Descriptive plots use the full dataset: they describe the data and are
    # not used for any model or hyperparameter choice.
    plots.class_balance(y)
    plots.correlation_heatmap(X)
    info["pca_components_for_95pct_variance"] = plots.pca_variance(X)
    plots.pca_scatter(X, y)

    # 2. Nested CV on the TRAIN split only ---------------------------------
    cv_results = evaluate.nested_cv(X_train, y_train)
    cv_results.to_csv(config.RESULTS_DIR / "nested_cv_results.csv", index=False)
    plots.metric_bars(cv_results, "f1", "05_f1_by_model_and_pca")
    plots.metric_bars(cv_results, "recall", "06_recall_by_model_and_pca")

    # 3. Select the best configuration using CV results only ---------------
    best = evaluate.best_configuration(cv_results)
    n_comp = None if best["n_components"] != best["n_components"] else int(best["n_components"])
    info["selected_by_nested_cv_on_train"] = {
        "model": best["model"], "features": best["features"],
        "f1": round(float(best["f1_mean"]), 4),
        "recall": round(float(best["recall_mean"]), 4)}
    print("\nSelected configuration:", info["selected_by_nested_cv_on_train"])

    # 4. Hold-out evaluation ------------------------------------------------
    # 4a. All models with the 30 original features (ROC curves, comparison)
    fitted_all, holdout_metrics, details = evaluate.holdout(X_train, X_test, y_train, y_test)
    holdout_metrics.to_csv(config.RESULTS_DIR / "holdout_metrics.csv", index=False)
    plots.roc_curves(y_test, details)

    # 4b. The selected configuration (confusion matrix and error analysis)
    if n_comp is None:
        fitted_sel, metrics_sel, details_sel = fitted_all, holdout_metrics, details
    else:
        fitted_sel, metrics_sel, details_sel = evaluate.holdout(
            X_train, X_test, y_train, y_test, n_components=n_comp)
    selected = metrics_sel[metrics_sel["model"] == best["model"]].copy()
    selected.insert(1, "features", best["features"])
    selected.to_csv(config.RESULTS_DIR / "holdout_selected.csv", index=False)
    plots.confusion_matrix_plot(details_sel[best["model"]]["confusion"],
                                f"{best['model']} / {best['features']}")
    wrong = evaluate.misclassified(X_test, y_test, details_sel[best["model"]]["y_pred"])
    wrong.to_csv(config.RESULTS_DIR / "misclassified_samples.csv")
    print(f"{len(wrong)} test samples misclassified by the selected configuration")

    # 4c. Decision regions in the PCA-2D plane (illustration only)
    fitted_2d, _, _ = evaluate.holdout(X_train, X_test, y_train, y_test, n_components=2)
    plots.decision_boundaries(fitted_2d, X_test, y_test)

    # 5. Save the summary used to write the article ------------------------
    info["best_params_selected"] = str(fitted_sel[best["model"]].best_params_)
    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("\nDone. Figures in ./figures, tables in ./results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke test")
    main(parser.parse_args().quick)