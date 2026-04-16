import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[TN, FP], [FN, TP]])

def accuracy(y_true, y_pred): #proporcion de predicciones correctas
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred): #de los predichos como positivos, cuantos realmente eran
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred): # cuantos positivos reales pudo detectar el modelo
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def roc_curve(y_true, y_proba): #trade off entre detectar positivos reales y generar falsos positivos
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
    
    fprs, tprs = np.array(fprs), np.array(tprs)
    orden = np.argsort(fprs)
    fprs, tprs = fprs[orden], tprs[orden]
    auc_val = np.trapz(tprs, fprs)
    return fprs, tprs, auc_val


def precision_recall_curve(y_true, y_proba): #precision vs recall
    thresholds = np.linspace(0, 1, 200)
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision(y_true, y_pred))
        recalls.append(recall(y_true, y_pred))

    precisions = np.array(precisions)
    recalls    = np.array(recalls)

    # Ordenar por recall creciente
    orden = np.argsort(recalls)
    recalls = recalls[orden]
    precisions = precisions[orden]

    auc_val = np.trapz(precisions, recalls)
    return precisions, recalls, auc_val


def f1(y_true, y_pred): #media armonica entre precision y recall
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def auc_roc(y_true, y_proba): #area bajo curva ROC, mide la capacidad discriminativa del modelo
    fprs, tprs, auc = roc_curve(y_true, y_proba)
    return auc  # integra tpr sobre fpr


def auc_pr(y_true, y_proba):
    precs, recs, auc = precision_recall_curve(y_true, y_proba)
    return auc  # integra precision sobre recall


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        "auc_roc": auc_roc(y_true, y_proba),
        "auc_pr": auc_pr(y_true, y_proba),
    }

def compute_metrics_multiclass(y_true, y_pred, y_proba):
    classes = np.unique(y_true)
    metrics = {}

    for k in classes:
        y_true_k = (y_true == k).astype(int)
        y_pred_k = (y_pred == k).astype(int)
        y_proba_k = y_proba[:, k]

        metrics[k] = compute_metrics(y_true_k, y_pred_k, y_proba_k)

    return metrics



