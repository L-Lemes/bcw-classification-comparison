# Breast Cancer Wisconsin: classification with KNN, Decision Tree, LDA, SVM and MLP

Final project for *Sistemas Inteligentes* (UFABC).
Group members: **<fill in the names here>**

## Goal
Classify breast tumours as **malignant** or **benign** from 30 numeric features
computed from cell-nucleus images. We compare six classifiers, with and without
PCA, and pay special attention to **recall of the malignant class**
(a missed malignant tumour is the costly error).

## Dataset
Breast Cancer Wisconsin (Diagnostic): 569 samples, 30 features, 357 benign and
212 malignant, no missing values. Loaded with `sklearn.datasets.load_breast_cancer`.
Original source: W. N. Wolberg, W. N. Street and O. L. Mangasarian, UCI Machine
Learning Repository (1995). Check the exact reference on the UCI page before citing.

## Method
- Labels are flipped so **malignant = 1** (positive class).
- Every model is a `Pipeline`: `StandardScaler -> (PCA) -> classifier`, so scaling and
  PCA are fitted on training folds only (no data leakage).
- Models: KNN, Decision Tree, LDA, SVM (linear and RBF), MLP.
- PCA settings: all 30 features, 2, 5 and 10 components.
- **Nested cross-validation** (5 outer folds, 3 inner folds, stratified): the inner
  `GridSearchCV` tunes hyperparameters for F1; the outer loop estimates performance.
- A stratified **hold-out split** (75/25) is used for ROC curves, confusion matrix,
  decision regions and error analysis.

## Project structure
```
bcw/config.py     experiment settings (seed, folds, PCA sizes, output folders)
bcw/data.py       loading, description and train/test split
bcw/models.py     models, hyperparameter grids, pipeline construction
bcw/evaluate.py   nested CV, hold-out evaluation, error analysis
bcw/plots.py      every figure used in the article
run_experiments.py   runs everything
```

## How to run (macOS, Apple Silicon)
Requires Python 3.15 (`python3.15 --version`). If missing:
`brew install python@3.15`. The setup script selects Python 3.15 explicitly.

Option A, one command:
```bash
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
python run_experiments.py          # full run, about 2 minutes
python run_experiments.py --quick  # fast smoke test
```

Option B, manual steps:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_experiments.py
```

Leave the environment with `deactivate`. To start over: `rm -rf .venv && ./setup.sh`.
Check that you are on a native build: `python -c "import platform; print(platform.machine())"` should print `arm64`.

## Outputs
- `figures/` : 01 class balance, 02 correlation heatmap, 03 PCA variance,
  04 PCA scatter, 05 F1 bars, 06 recall bars, 07 confusion matrix,
  08 ROC curves, 09 decision regions in PCA-2D.
- `results/nested_cv_results.csv` : mean and std of every metric per model and PCA setting.
- `results/holdout_metrics.csv` : hold-out metrics and best hyperparameters.
- `results/misclassified_samples.csv` : test samples the best model got wrong.
- `results/summary.json` : dataset facts and best configuration.

## Reproducibility
Everything uses `RANDOM_STATE = 42` (see `bcw/config.py`).
Results may vary slightly across scikit-learn versions.
