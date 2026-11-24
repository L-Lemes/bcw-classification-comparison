"""Central configuration: change experiment settings here only."""
from pathlib import Path

# Reproducibility
RANDOM_STATE = 42

# Hold-out split used for ROC curves, confusion matrix and decision boundaries
TEST_SIZE = 0.25

# Nested cross-validation (outer = performance estimate, inner = hyperparameter search)
OUTER_FOLDS = 5
INNER_FOLDS = 3

# PCA settings compared for every model: None = all 30 original features
PCA_SETTINGS = [None, 2, 5, 10]

# Metric optimised during hyperparameter search.
# Malignant is the positive class, so F1 balances missed tumours and false alarms.
SEARCH_SCORING = "f1"

# Output folders (created automatically)
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
