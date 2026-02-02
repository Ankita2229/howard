"""
Generate curve/line plot comparison figures for all metrics across all 7 models
Replaces bar charts with line graphs as requested
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the external validation performance data
external_df = pd.read_csv('final_results/external_validation_performance.csv')

# Define models and metrics
models = external_df['Model'].tolist()
metrics = ['AUROC', 'PR AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_labels = {
    'AUROC': 'AUROC',
    'PR AUC': 'PR AUC',
    'Accuracy': 'Accuracy',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1 Score': 'F1 Score'
}

# Create output directory
output_dir = Path('final_results/additional_figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Color palette for models
colors = plt.cm.tab10(np.linspace(0, 0.7, len(models)))

print("Generating curve comparison figures...")

# 1. Individual metric line plots
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 8))

    values = external_df[metric].values
    x_positions = np.arange(len(models))

    # Plot line
    ax.plot(x_positions, values, marker='o', linewidth=3, markersize=12,
            color='#2E86AB', alpha=0.8, label=metric_labels[metric])

    # Add value labels on points
    for i, (x, y) in enumerate(zip(x_positions, values)):
        ax.annotate(f'{y:.4f}',
                   xy=(x, y),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=10,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Styling
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_labels[metric], fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_labels[metric]} Comparison Across All Models\n(External Validation)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set y-axis limits for better visualization
    y_min = max(0, values.min() - 0.05)
    y_max = min(1, values.max() + 0.05)
    ax.set_ylim(y_min, y_max)

    # Add reference line for best performance
    best_val = values.max()
    ax.axhline(y=best_val, color='green', linestyle='--', alpha=0.3, linewidth=2)

    plt.tight_layout()

    # Save with appropriate filename
    filename_map = {
        'AUROC': 'auroc_comparison.png',
        'PR AUC': 'pr_auc_comparison.png',
        'Accuracy': 'accuracy_comparison.png',
        'Precision': 'precision_comparison.png',
        'Recall': 'recall_comparison.png',
        'F1 Score': 'f1_score_comparison.png'
    }

    output_path = output_dir / filename_map[metric]
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# 2. Comprehensive multi-metric line plot (all metrics on one graph)
print("\nGenerating comprehensive multi-metric curve plot...")
fig, ax = plt.subplots(figsize=(14, 9))

x_positions = np.arange(len(models))

# Plot each metric as a separate line
for i, metric in enumerate(metrics):
    values = external_df[metric].values
    ax.plot(x_positions, values, marker='o', linewidth=2.5, markersize=10,
           label=metric_labels[metric], alpha=0.8)

# Styling
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Comprehensive Performance Metrics Comparison\n(All Metrics, External Validation)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(output_dir / 'comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'comprehensive_metrics_comparison.png'}")
plt.close()

# 3. Performance heatmap (keep this as is - it's not a bar chart)
print("\nGenerating performance heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data for heatmap
heatmap_data = external_df[metrics].T
heatmap_data.columns = models

# Create heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn',
           cbar_kws={'label': 'Score'}, linewidths=0.5,
           vmin=0.5, vmax=1.0, ax=ax)

ax.set_title('Performance Heatmap: All Metrics × All Models\n(External Validation)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'performance_heatmap.png'}")
plt.close()

# 4. Model ranking heatmap (keep this as is)
print("\nGenerating model ranking heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ranks (1 = best, 7 = worst)
ranking_data = external_df[metrics].T.rank(axis=1, ascending=False)
ranking_data.columns = models

# Create heatmap
sns.heatmap(ranking_data, annot=True, fmt='.0f', cmap='RdYlGn_r',
           cbar_kws={'label': 'Rank (1=Best)'}, linewidths=0.5,
           vmin=1, vmax=7, ax=ax)

ax.set_title('Model Ranking Heatmap: Performance Across Metrics\n(1=Best, 7=Worst)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(output_dir / 'model_ranking_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'model_ranking_heatmap.png'}")
plt.close()

# 5. Radar chart for top 3 models
print("\nGenerating radar chart for top 3 models...")

# Calculate average score for each model
external_df['Average'] = external_df[metrics].mean(axis=1)
top3 = external_df.nlargest(3, 'Average')

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Number of metrics
num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot each model
colors_top3 = ['#2E86AB', '#A23B72', '#F18F01']
for idx, (_, row) in enumerate(top3.iterrows()):
    values = row[metrics].tolist()
    values += values[:1]  # Complete the circle

    ax.plot(angles, values, 'o-', linewidth=2.5,
           label=row['Model'], color=colors_top3[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_top3[idx])

# Styling
ax.set_xticks(angles[:-1])
ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_title('Radar Chart: Top 3 Models Performance\n(All Metrics, External Validation)',
            fontsize=16, fontweight='bold', pad=30, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'radar_chart_top3.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'radar_chart_top3.png'}")
plt.close()

print("\n" + "="*60)
print("✓ All curve comparison figures generated successfully!")
print(f"✓ Location: {output_dir}")
print("="*60)
