import numpy as np
import matplotlib.pyplot as plt

#ver si me conviene hacer una clase para metricas y hacer metodos para cada una

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[TN, FP], [FN, TP]])

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def roc_curve(y_true, y_proba):
    thresholds = np.linspace(0, 1, 200)
    fprs, tprs = [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        tprs.append(TP / (TP + FN) if (TP + FN) > 0 else 0.0)
        fprs.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
    return np.array(fprs), np.array(tprs)

def auc(x, y):
    order = np.argsort(x)
    return np.trapz(y[order], x[order])

def precision_recall_curve(y_true, y_proba):
    thresholds = np.linspace(0, 1, 200)
    precisions = []
    recalls = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision(y_true, y_pred))
        recalls.append(recall(y_true, y_pred))
    return np.array(precisions), np.array(recalls)

def plot_roc(y_true, y_proba, title=''):
    fprs, tprs = roc_curve(y_true, y_proba)
    auc_roc = auc(fprs, tprs)

    precs, recs = precision_recall_curve(y_true, y_proba)
    auc_pr = auc(recs, precs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    axes[0].plot(fprs, tprs, color='steelblue', lw=2, label=f'AUC-ROC = {auc_roc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title(f'Curva ROC {title}')
    axes[0].legend()

    axes[1].plot(recs, precs, color='steelblue', lw=2, label=f'AUC-PR = {auc_pr:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Curva PR {title}')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def auc_roc(y_true, y_proba):
    fprs, tprs = roc_curve(y_true, y_proba)
    return auc(fprs, tprs)  # integra tpr sobre fpr


def auc_pr(y_true, y_proba):
    precs, recs = precision_recall_curve(y_true, y_proba)
    return auc(recs, precs)  # integra precision sobre recall

def plot_confusion_matrix(y_true, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    return ax

def plot_pr(y_true, y_proba, title='', ax=None):
    if ax is None:
        _, ax = plt.subplots()
    recalls, precisions = precision_recall_curve(y_true, y_proba)
    auc = auc_pr(y_true, y_proba)
    baseline = y_true.mean()
    ax.plot(recalls, precisions, label=f"AUC-PR = {auc:.3f}")
    ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve {title}")
    ax.legend()
    return ax


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy"  : accuracy(y_true, y_pred),
        "precision" : precision(y_true, y_pred),
        "recall"    : recall(y_true, y_pred),
        "f1"        : f1(y_true, y_pred),
        "auc_roc"   : auc_roc(y_true, y_proba),
        "auc_pr"    : auc_pr(y_true, y_proba),
    }

