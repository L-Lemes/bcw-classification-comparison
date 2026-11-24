"""All figures used in the article and slides (saved as PNG, 300 dpi).

All text drawn inside the figures is in Brazilian Portuguese. Internal names
(model names, feature names, PCA labels) stay in English in the code and CSV
files and are translated only when drawn, through the helpers below.
"""
import matplotlib

matplotlib.use("Agg")  # no GUI needed; works on any machine
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from . import config

COLORS = {0: "#2a7fba", 1: "#d6452f"}  # benign = blue, malignant = red
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- translation
DISPLAY_NAMES = {
    "All 30 features": "30 atributos",
    "Decision Tree": "Árvore de Decisão",
}

FEATURE_WORDS = {
    "radius": "raio",
    "texture": "textura",
    "perimeter": "perímetro",
    "area": "área",
    "smoothness": "suavidade",
    "compactness": "compacidade",
    "concavity": "concavidade",
    "concave points": "pontos côncavos",
    "symmetry": "simetria",
    "fractal dimension": "dimensão fractal",
}

# Prefix/suffix used by scikit-learn: "mean X", "X error", "worst X"
FEATURE_KIND = {"mean": "média", "error": "erro", "worst": "pior"}


def tr(text):
    """Translate a model name or PCA label for display."""
    for english, portuguese in DISPLAY_NAMES.items():
        text = text.replace(english, portuguese)
    return text


def tr_feature(name):
    """Translate a feature name such as 'mean radius' -> 'raio (média)'."""
    if name.startswith("mean "):
        kind, base = "mean", name[len("mean "):]
    elif name.startswith("worst "):
        kind, base = "worst", name[len("worst "):]
    elif name.endswith(" error"):
        kind, base = "error", name[:-len(" error")]
    else:
        return name
    return f"{FEATURE_WORDS.get(base, base)} ({FEATURE_KIND[kind]})"


def num(value, digits=3):
    """Format a number with a decimal comma (Brazilian convention)."""
    return f"{value:.{digits}f}".replace(".", ",")


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.name}")


# -------------------------------------------------------------------- figures
def class_balance(y):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    counts = [np.sum(y == 0), np.sum(y == 1)]
    bars = ax.bar(["Benigno", "Maligno"], counts,
                  color=[COLORS[0], COLORS[1]])
    ax.bar_label(bars)
    ax.set_ylabel("Amostras")
    ax.set_title("Distribuição das classes")
    _save(fig, "01_class_balance")


def correlation_heatmap(X):
    corr = X.corr().to_numpy()
    labels = [tr_feature(c) for c in X.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(X.shape[1]), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(X.shape[1]), labels, fontsize=6)
    fig.colorbar(im, ax=ax, label="Correlação de Pearson")
    ax.set_title("Correlação entre atributos (muitos são redundantes)")
    _save(fig, "02_correlation_heatmap")


def pca_variance(X):
    Z = StandardScaler().fit_transform(X)
    ratio = PCA().fit(Z).explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(range(1, len(ratio) + 1), cumulative, marker="o", ms=3)
    ax.axhline(0.95, ls="--", color="gray", lw=1)
    n95 = int(np.argmax(cumulative >= 0.95)) + 1
    ax.annotate(f"{n95} componentes → 95%", (n95, 0.95),
                xytext=(n95 + 3, 0.80), arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("Número de componentes principais")
    ax.set_ylabel("Variância explicada acumulada")
    ax.set_title("Variância explicada pelo PCA")
    _save(fig, "03_pca_variance")
    return n95


def pca_scatter(X, y):
    Z = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(
        StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, name in [(0, "Benigno"), (1, "Maligno")]:
        ax.scatter(Z[y == label, 0], Z[y == label, 1], s=12, alpha=0.7,
                   color=COLORS[label], label=name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Amostras projetadas nas duas primeiras componentes")
    ax.legend()
    _save(fig, "04_pca_scatter")


BAR_TITLES = {
    "f1": "F1-score para tumores malignos",
    "recall": "Recall para tumores malignos "
              "(tumores não detectados são o erro mais grave)",
}
METRIC_NAMES = {"f1": "F1", "recall": "Recall", "auc": "AUC"}


def metric_bars(cv_results, metric, filename, title=None):
    """Grouped bars: one group per model, one bar per PCA setting, error = std.

    `title` is kept for compatibility with older callers; the Portuguese title
    comes from BAR_TITLES when the metric is known.
    """
    title = BAR_TITLES.get(metric, title)
    models = list(dict.fromkeys(cv_results["model"]))
    settings = list(dict.fromkeys(cv_results["features"]))
    width = 0.8 / len(settings)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, setting in enumerate(settings):
        sub = cv_results[cv_results["features"] == setting].set_index("model")
        means = [sub.loc[m, f"{metric}_mean"] for m in models]
        stds = [sub.loc[m, f"{metric}_std"] for m in models]
        ax.bar(np.arange(len(models)) + i * width, means, width, yerr=stds,
               capsize=2, label=tr(setting))
    ax.set_xticks(np.arange(len(models)) + width * (len(settings) - 1) / 2,
                  [tr(m) for m in models])
    ax.set_ylim(0.70, 1.02)
    ax.set_ylabel(f"{METRIC_NAMES.get(metric, metric)} (média ± desvio padrão, "
                  f"VC aninhada de {config.OUTER_FOLDS} folds)")
    ax.set_title(title, pad=28)
    # Legend above the plot area so it never hides bars
    ax.legend(ncol=len(settings), fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False)
    _save(fig, filename)


def confusion_matrix_plot(cm, model_name, filename="07_confusion_matrix"):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Benigno", "Maligno"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, value, ha="center", va="center", fontsize=14,
                color="white" if value > cm.max() / 2 else "black")
    ax.set_title(f"{tr(model_name)}\n(conjunto de teste hold-out)")
    _save(fig, filename)


def roc_curves(y_test, details):
    """ROC curves: full view (left) and zoom on the top-left corner (right).

    The AUC of each model is shown in the legend. The zoom panel is needed
    because most curves overlap near the top-left corner.
    """
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(10, 4.4))
    for name, d in details.items():
        fpr, tpr, _ = roc_curve(y_test, d["y_score"])
        auc = roc_auc_score(y_test, d["y_score"])
        for target in (ax, zoom):
            target.plot(fpr, tpr, lw=1.6, label=f"{tr(name)} (AUC = {num(auc)})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_title("Curvas ROC (hold-out, 30 atributos)")
    ax.legend(fontsize=8, loc="lower right")
    zoom.set_xlim(0, 0.25)
    zoom.set_ylim(0.85, 1.01)
    zoom.set_title("Zoom no canto superior esquerdo")
    for target in (ax, zoom):
        target.set_xlabel("Taxa de falsos positivos")
        target.set_ylabel("Taxa de verdadeiros positivos (recall)")
    fig.tight_layout()
    _save(fig, "08_roc_curves")


def decision_boundaries(fitted_pca2, X_test, y_test):
    """Decision regions in the PCA-2D space, one subplot per model.

    `fitted_pca2` holds GridSearchCV objects trained on a PCA-2 pipeline, so
    everything before the classifier maps samples into the 2D plane.
    """
    names = list(fitted_pca2)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.ravel(), names):
        pipe = fitted_pca2[name].best_estimator_
        Z = pipe[:-1].transform(X_test)  # scaler + PCA
        clf = pipe[-1]
        pad = 0.5
        xx, yy = np.meshgrid(
            np.linspace(Z[:, 0].min() - pad, Z[:, 0].max() + pad, 300),
            np.linspace(Z[:, 1].min() - pad, Z[:, 1].max() + pad, 300))
        grid = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, grid, alpha=0.25, levels=[-0.5, 0.5, 1.5],
                    colors=[COLORS[0], COLORS[1]])
        for label in (0, 1):
            ax.scatter(Z[y_test == label, 0], Z[y_test == label, 1], s=10,
                       color=COLORS[label])
        ax.set_title(tr(name))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("Regiões de decisão no espaço PCA-2D "
                 "(amostras de teste; azul = benigno, vermelho = maligno)")
    fig.tight_layout()
    _save(fig, "09_decision_boundaries")