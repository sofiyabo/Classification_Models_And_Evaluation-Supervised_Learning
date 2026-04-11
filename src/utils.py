import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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