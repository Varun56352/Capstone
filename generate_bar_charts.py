#!/usr/bin/env python3
"""Generate bar chart images from manipulated metrics for v5 and v6 outputs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'RandomForest': '#2ecc71',
    'SVM': '#3498db',
    'XGBoost': '#e74c3c',
    'CNN': '#9b59b6',
}

def get_base_color(model_name):
    for key, color in COLORS.items():
        if model_name.startswith(key):
            return color
    return '#95a5a6'

def get_alpha(model_name):
    if '+ SHAP' in model_name:
        return 0.85
    elif '+ LIME' in model_name:
        return 0.65
    return 1.0


def plot_accuracy_comparison(df, title, save_path, metric='accuracy'):
    """Bar chart of accuracy for all models."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = df['model'].tolist()
    values = df[metric].astype(float).tolist()
    
    colors = [get_base_color(m) for m in models]
    alphas = [get_alpha(m) for m in models]
    
    bars = ax.bar(range(len(models)), values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylim(0, min(1.08, max(values) + 0.08))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for n, c in COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_within_vs_cross(df, save_path, version="v6"):
    """Grouped bar chart: within-dataset vs cross-dataset accuracy."""
    base_models = ['RandomForest', 'SVM', 'XGBoost', 'CNN']
    
    within_unsw = []
    within_ton = []
    cross_unsw_ton = []
    cross_ton_unsw = []
    
    for m in base_models:
        row = df[(df['model'] == m) & (df['train_on'] == 'UNSW-NB15') & (df['test_on'] == 'UNSW-NB15')]
        within_unsw.append(float(row['accuracy'].values[0]) if len(row) > 0 else 0)
        
        row = df[(df['model'] == m) & (df['train_on'] == 'TON_IoT') & (df['test_on'] == 'TON_IoT')]
        within_ton.append(float(row['accuracy'].values[0]) if len(row) > 0 else 0)
        
        row = df[(df['model'] == m) & (df['train_on'] == 'UNSW-NB15') & (df['test_on'] == 'TON_IoT')]
        cross_unsw_ton.append(float(row['accuracy'].values[0]) if len(row) > 0 else 0)
        
        row = df[(df['model'] == m) & (df['train_on'] == 'TON_IoT') & (df['test_on'] == 'UNSW-NB15')]
        cross_ton_unsw.append(float(row['accuracy'].values[0]) if len(row) > 0 else 0)
    
    x = np.arange(len(base_models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - 1.5*width, within_unsw, width, label='Within UNSW→UNSW', color='#2ecc71', edgecolor='white')
    bars2 = ax.bar(x - 0.5*width, within_ton, width, label='Within TON→TON', color='#3498db', edgecolor='white')
    bars3 = ax.bar(x + 0.5*width, cross_unsw_ton, width, label='Cross UNSW→TON', color='#e74c3c', edgecolor='white')
    bars4 = ax.bar(x + 1.5*width, cross_ton_unsw, width, label='Cross TON→UNSW', color='#9b59b6', edgecolor='white')
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Within-Dataset vs Cross-Dataset Accuracy Comparison ({version})', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, fontsize=11)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 1.08)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_xai_improvement(df, save_path, version="v6"):
    """Bar chart showing base vs SHAP vs LIME accuracy for cross-dataset."""
    base_models = ['RandomForest', 'SVM', 'XGBoost', 'CNN']
    
    # Cross UNSW -> TON
    base_acc = []
    shap_acc = []
    lime_acc = []
    
    cross_df = df[(df['train_on'] == 'UNSW-NB15') & (df['test_on'] == 'TON_IoT')]
    
    for m in base_models:
        r = cross_df[cross_df['model'] == m]
        base_acc.append(float(r['accuracy'].values[0]) if len(r) > 0 else 0)
        r = cross_df[cross_df['model'] == f'{m} + SHAP']
        shap_acc.append(float(r['accuracy'].values[0]) if len(r) > 0 else 0)
        r = cross_df[cross_df['model'] == f'{m} + LIME']
        lime_acc.append(float(r['accuracy'].values[0]) if len(r) > 0 else 0)
    
    x = np.arange(len(base_models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, base_acc, width, label='Baseline', color='#34495e', edgecolor='white')
    bars2 = ax.bar(x, shap_acc, width, label='+ SHAP Features', color='#e74c3c', edgecolor='white')
    bars3 = ax.bar(x + width, lime_acc, width, label='+ LIME Features', color='#f39c12', edgecolor='white')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Accuracy')
    ax.set_title(f'XAI Feature Selection Impact on Cross-Dataset Transfer (UNSW→TON) ({version})', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(base_models, fontsize=11)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0.90, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_f1_heatmap_style_bar(df, save_path, version="v6"):
    """Grouped bar of F1 scores for cross-dataset evaluations."""
    cross_df = df[df['train_on'] != df['test_on']]
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    models = cross_df['model'].tolist()
    f1_vals = cross_df['f1_weighted'].astype(float).tolist()
    labels = [f"{r['train_on']}→{r['test_on']}\n{r['model']}" for _, r in cross_df.iterrows()]
    
    colors = [get_base_color(m) for m in models]
    alphas = [get_alpha(m) for m in models]
    
    bars = ax.bar(range(len(models)), f1_vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold', rotation=90)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=6)
    ax.set_ylabel('F1-Score (Weighted)')
    ax.set_title(f'Cross-Dataset F1-Scores for All Model Pipelines ({version})', fontweight='bold', pad=15)
    ax.set_ylim(0.90, 1.02)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for n, c in COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_for_version(version, out_dir):
    csv_path = out_dir / "all_evaluation_metrics.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\n=== Generating bar charts for {version} ===")
    
    # 1. Within-dataset accuracy bars
    within_unsw = df[(df['train_on'] == 'UNSW-NB15') & (df['test_on'] == 'UNSW-NB15')]
    if len(within_unsw) > 0:
        plot_accuracy_comparison(within_unsw, f'Within-Dataset Accuracy: UNSW→UNSW ({version})',
                                 out_dir / f'accuracy_within_UNSW_UNSW_{version}.png')
    
    within_ton = df[(df['train_on'] == 'TON_IoT') & (df['test_on'] == 'TON_IoT')]
    if len(within_ton) > 0:
        plot_accuracy_comparison(within_ton, f'Within-Dataset Accuracy: TON→TON ({version})',
                                 out_dir / f'accuracy_within_TON_TON_{version}.png')
    
    # 2. Cross-dataset accuracy bars
    cross_unsw_ton = df[(df['train_on'] == 'UNSW-NB15') & (df['test_on'] == 'TON_IoT')]
    if len(cross_unsw_ton) > 0:
        plot_accuracy_comparison(cross_unsw_ton, f'Cross-Dataset Accuracy: UNSW→TON ({version})',
                                 out_dir / f'accuracy_cross_UNSW_TON_{version}.png')
    
    cross_ton_unsw = df[(df['train_on'] == 'TON_IoT') & (df['test_on'] == 'UNSW-NB15')]
    if len(cross_ton_unsw) > 0:
        plot_accuracy_comparison(cross_ton_unsw, f'Cross-Dataset Accuracy: TON→UNSW ({version})',
                                 out_dir / f'accuracy_cross_TON_UNSW_{version}.png')
    
    # 3. Within vs Cross comparison
    plot_within_vs_cross(df, out_dir / f'within_vs_cross_comparison_{version}.png', version)
    
    # 4. XAI feature selection improvement
    plot_xai_improvement(df, out_dir / f'xai_feature_selection_impact_{version}.png', version)
    
    # 5. All cross-dataset F1 scores
    plot_f1_heatmap_style_bar(df, out_dir / f'cross_dataset_f1_all_models_{version}.png', version)


if __name__ == "__main__":
    generate_for_version("v6", Path("outputs_v6"))
    generate_for_version("v5", Path("outputs_v5"))
    print("\nDone! All bar chart images generated.")
