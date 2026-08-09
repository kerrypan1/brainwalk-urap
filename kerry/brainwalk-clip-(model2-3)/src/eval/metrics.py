from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def classification_report_dict(y_true, y_pred, classes=(0, 1, 2, 3)) -> dict:
    """Score raw predictions, rounding only for classification metrics.

    MAE is calculated directly on the continuous predictions. Accuracy, F1,
    balanced accuracy, QWK, and the confusion matrix use predictions rounded
    and clipped to the supplied ordinal class range.
    """
    y_true = np.asarray(y_true)
    y_raw = np.asarray(y_pred, dtype=float)
    y_class = np.clip(np.rint(y_raw), min(classes), max(classes)).astype(int)
    per_class_f1 = f1_score(
        y_true, y_class, labels=list(classes), average=None, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_class)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_class)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_class,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_class,
                labels=list(classes),
                average="weighted",
                zero_division=0,
            )
        ),
        "mae": float(np.mean(np.abs(y_true - y_raw))),
        "qwk": float(
            cohen_kappa_score(
                y_true, y_class, labels=list(classes), weights="quadratic"
            )
        ),
        "per_class_f1": {int(c): float(v) for c, v in zip(classes, per_class_f1)},
        "confusion": confusion_matrix(y_true, y_class, labels=list(classes)).tolist(),
    }


def fold_classification_report(
    y_true, y_pred, folds, classes=(0, 1, 2, 3)
) -> dict:
    """Equal-weight mean and sample SD of held-out fold metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    folds = np.asarray(folds)
    labels = sorted(np.unique(folds))
    reports = []
    for label in labels:
        mask = folds == label
        row = classification_report_dict(y_true[mask], y_pred[mask], classes)
        row["fold"] = str(label)
        reports.append(row)

    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mae",
        "qwk",
    )
    result = {"n": int(len(y_true)), "n_folds": len(labels)}
    result["sd"] = {}
    for key in metric_keys:
        values = np.asarray([row[key] for row in reports], dtype=float)
        result[key] = float(np.nanmean(values))
        result["sd"][key] = float(np.nanstd(values, ddof=1))

    result["fold_metrics"] = reports
    result["pooled"] = classification_report_dict(y_true, y_pred, classes)
    return result


def fold_constant_baselines(y_true, folds, classes=(0, 1, 2, 3)) -> dict:
    """Fit mean, median, and mode on each outer training fold."""
    y_true = np.asarray(y_true, dtype=float)
    folds = np.asarray(folds)
    predictions = {
        "mean": np.full(len(y_true), np.nan),
        "median": np.full(len(y_true), np.nan),
        "mode": np.full(len(y_true), np.nan),
    }
    fitted_values = {name: {} for name in predictions}

    for label in sorted(np.unique(folds)):
        test = folds == label
        train_y = y_true[~test]
        values = {
            "mean": float(np.mean(train_y)),
            "median": float(np.median(train_y)),
            "mode": float(np.bincount(train_y.astype(int)).argmax()),
        }
        for name, value in values.items():
            predictions[name][test] = value
            fitted_values[name][str(label)] = value

    return {
        name: {
            **fold_classification_report(y_true, pred, folds, classes),
            "fitted_values": fitted_values[name],
        }
        for name, pred in predictions.items()
    }


def constant_baselines(y_true, classes=(0, 1, 2, 3)) -> dict:
    """Reference metrics for trivial constant predictors (mode / median-int).

    On imbalanced ordinal data these are the real bars to clear: 'always mode'
    maximizes accuracy, 'always median' tends to minimize MAE.
    """
    y_true = np.asarray(y_true)
    mode = int(np.bincount(y_true).argmax())
    median = int(np.median(y_true))
    out = {}
    for name, c in [(f"always_{mode}(mode)", mode), (f"always_{median}(median)", median)]:
        pred = np.full_like(y_true, c)
        r = classification_report_dict(y_true, pred, classes=classes)
        out[name] = {k: r[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "mae", "qwk")}
    return out


def bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=2000, seed=0,
                 classes=(0, 1, 2, 3)):
    """Percentile bootstrap 95% CI for a metric over paired (y_true, y_pred)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = classification_report_dict(y_true[idx], y_pred[idx], classes=classes)
        vals.append(r[metric])
    vals = np.asarray(vals)
    point = classification_report_dict(y_true, y_pred, classes=classes)[metric]
    return {"point": float(point), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def save_confusion_png(y_true, y_pred, path, classes=(0, 1, 2, 3), title="Confusion"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_class = np.clip(
        np.rint(np.asarray(y_pred, dtype=float)), min(classes), max(classes)
    ).astype(int)
    cm = confusion_matrix(y_true, y_class, labels=list(classes))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
