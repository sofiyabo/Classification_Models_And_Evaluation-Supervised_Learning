import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import preprocessing as prep
import data_splitting as data
import metrics as mtr

def boxplots(df, feats):
    c = len(feats) // 2 +1
    fig, axes = plt.subplots(2, c, figsize=(16, 8), dpi= 150)
    for ax, col in zip(axes.flatten(), feats):
        sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(col)
    plt.suptitle("Boxplots")
    plt.tight_layout()
    plt.show()
    return

def scatters(df, feats):
    c = len(feats) // 2 +1
    fig, axes = plt.subplots(2, c, figsize=(16, 8))
    for ax, col in zip(axes.flatten(), feats):
        sns.scatterplot(data=df, x=col, y='rendimiento', alpha=0.4, ax=ax)
        ax.set_title(f'{col} vs rendimiento')
    plt.suptitle("Scatterplots")
    plt.tight_layout()
    plt.show()

def plot_school_analysis(df, feats_numericas, label_rend, label_sem):
    
    # ── 1. Boxplots de cada feature numérica por escuela ──────────────────────
    c = 3
    r = len(feats_numericas) // c + (1 if len(feats_numericas) % c else 0)
    fig, axes = plt.subplots(r, c, figsize=(18, r * 4), dpi=120)
    axes = axes.flatten()

    for i, col in enumerate(feats_numericas):
        sns.boxplot(data=df, x='escuela', y=col, ax=axes[i], color='steelblue')
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Escuela')
        axes[i].set_ylabel('')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribución de features por escuela', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ── 2. Heatmap de correlación feature-target por escuela ──────────────────
    corrs = df.groupby('escuela').apply(
        lambda g: g[feats_numericas].corrwith(g['rendimiento'])
    )

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    sns.heatmap(corrs.T, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, ax=ax)
    ax.set_title('Correlación feature-target por escuela', fontsize=14, fontweight='bold')
    ax.set_xlabel('Escuela')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # ── 3. Rendimiento promedio por escuela y semestre ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    prop_escuela = df.groupby('escuela')['rendimiento'].mean().reset_index()
    sns.barplot(data=prop_escuela, x='escuela', y='rendimiento',
                ax=axes[0], color='steelblue')
    axes[0].set_title('Rendimiento promedio por escuela', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Rendimiento promedio')
    axes[0].set_xlabel('Escuela')
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(['Insuficiente', 'Regular', 'Bueno', 'Excelente'])
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f'{bar.get_height():.2f}', ha='center', fontsize=9)

    prop_semestre = df.groupby('semestre')['rendimiento'].mean().reset_index()
    prop_semestre['semestre_label'] = prop_semestre['semestre'].map(label_sem)
    sns.barplot(data=prop_semestre, x='semestre_label', y='rendimiento',
                ax=axes[1], color='steelblue')
    axes[1].set_title('Rendimiento promedio por semestre', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Rendimiento promedio')
    axes[1].set_xlabel('Semestre')
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['Insuficiente', 'Regular', 'Bueno', 'Excelente'])
    axes[1].tick_params(axis='x', rotation=30)
    for bar in axes[1].patches:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f'{bar.get_height():.2f}', ha='center', fontsize=9)

    plt.suptitle('Rendimiento promedio por grupo', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ── 4. Distribución multiclase por escuela (stacked barplot) ──────────────
    df = df.copy()
    df['rendimiento_label'] = df['rendimiento'].map(label_rend)

    order = ['Insuficiente', 'Regular', 'Bueno', 'Excelente']
    prop_multi = (df.groupby(['escuela', 'rendimiento_label'])
                    .size()
                    .unstack(fill_value=0))
    prop_multi = prop_multi[order].div(prop_multi.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    prop_multi.plot(kind='bar', stacked=True, ax=ax,
                    colormap='RdYlGn', edgecolor='white')
    ax.set_title('Distribución de clases multiclase por escuela',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Escuela')
    ax.set_ylabel('Proporción')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Rendimiento', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def cv(folds, model_fn, preprocess_fn, metrics_fn):
    all_metrics = []

    for fold in folds:
        df_train, df_val = fold

        X_train, y_train, X_val, y_val = preprocess_fn(df_train, df_val)
        model   = model_fn(X_train, y_train)
        metrics = metrics_fn(model, X_val, y_val)
        all_metrics.append(metrics)

    return all_metrics

def matrices(df, feature_cols, target):
    
    X = df[feature_cols].values
    y = df[target].values
    return X, y


def pipeline(df_train, df_val, feats_imp, exclude):

    meds = prep.medians(df_train, feats_imp)
    df_train = prep.impute(df_train, meds, feats_imp)
    df_val = prep.impute(df_val, meds, feats_imp)
    params = prep.params_norm(df_train, exclude)
    df_train = prep.normalize_df(df_train, params)
    df_val = prep.normalize_df(df_val, params)
    return df_train, df_val



def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", ax=None):
    if ax is None:
        _, ax = plt.subplots()
    cm = mtr.confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax

def plot_pr(y_true, y_proba, title='', ax=None):
    if ax is None:
        _, ax = plt.subplots()
    recalls, precisions, auc = mtr.precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()
    ax.plot(recalls, precisions, label=f"AUC-PR = {auc:.3f}")
    ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve {title}")
    ax.legend()
    return ax

def plot_roc(y_true, y_proba, title=''):
    fprs, tprs, auc_roc = mtr.roc_curve(y_true, y_proba)

    precs, recs, auc_pr = mtr.precision_recall_curve(y_true, y_proba)

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


def plot_roc_multiclass(y_true, y_proba, classes, class_names, title="ROC Multiclase"):
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    
    for ax, k in zip(axes, classes):
        y_true_k  = (y_true == k).astype(int)
        y_proba_k = y_proba[:, k]
        
        fpr, tpr, auc= mtr.roc_curve(y_true_k, y_proba_k)
        
        ax.plot(fpr, tpr, color='steelblue', label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title(f"{class_names[k]}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_pr_multiclass(y_true, y_proba, classes, class_names, title="PR Multiclase"):
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    
    for ax, k in zip(axes, classes):
        y_true_k = (y_true == k).astype(int)
        y_proba_k = y_proba[:, k]
        
        prec, rec, auc = mtr.precision_recall_curve(y_true_k, y_proba_k)

        baseline = y_true_k.mean()
        
        ax.plot(rec, prec, color='steelblue', label=f"AUC = {auc:.3f}")
        ax.axhline(baseline, linestyle='--', color='gray', 
                   alpha=0.5, label=f"Baseline = {baseline:.3f}")
        ax.set_title(f"{class_names[k]}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_multiclass(y_true, y_pred, class_names, title="Confusion Matrix"):
    classes = sorted(np.unique(y_true))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    
    for i, real in enumerate(classes):
        for j, pred in enumerate(classes):
            cm[i, j] = np.sum((y_true == real) & (y_pred == pred))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    
    labels = [class_names[k] for k in classes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.show()