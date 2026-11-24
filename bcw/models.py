"""Model definitions, hyperparameter grids and pipeline construction.

Every model lives inside a Pipeline: StandardScaler -> (optional PCA) -> classifier.
Scaling and PCA are fitted only on training folds, so there is no data leakage.
"""
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config

SEED = config.RANDOM_STATE


def model_specs():
    """Map model name -> (classifier, hyperparameter grid).

    Grid keys use the "clf__" prefix because the classifier is the pipeline
    step named "clf".
    """
    return {
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 11, 15],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [2, 3, 4, 5, None],
             "clf__min_samples_leaf": [1, 5, 10]},
        ),
        "LDA": (
            LinearDiscriminantAnalysis(),
            {},  # no hyperparameters to tune
        ),
        "SVM (linear)": (
            SVC(kernel="linear", random_state=SEED),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", random_state=SEED),
            {"clf__C": [1, 10, 100],
             "clf__gamma": ["scale", 0.01, 0.1]},
        ),
        "MLP": (
            MLPClassifier(max_iter=2000, random_state=SEED),
            {"clf__hidden_layer_sizes": [(16,), (32,), (32, 16)],
             "clf__alpha": [1e-4, 1e-2]},
        ),
    }


def build_pipeline(name, n_components=None):
    """Build the pipeline and grid for one model and one PCA setting.

    Parameters
    ----------
    name : model name from `model_specs()`
    n_components : number of PCA components, or None to use all features
    """
    clf, grid = model_specs()[name]
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps), grid


def pca_label(n_components):
    """Readable label for a PCA setting."""
    return "All 30 features" if n_components is None else f"PCA-{n_components}"
