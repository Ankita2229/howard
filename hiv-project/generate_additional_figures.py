"""
Generate Additional Comparison Figures for All Models
======================================================
This script creates individual metric comparison charts that were missing:
- AUROC comparison (bar chart)
- F1 Score comparison (bar chart)
- Precision comparison (bar chart)
- Recall comparison (bar chart)
- Accuracy comparison (bar chart)
- PR AUC comparison (bar chart)
- Comprehensive multi-metric comparison (grouped bar chart)
- Metric heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
os.makedirs('final_results/additional_figures', exist_ok=True)

print("="*80)
print("GENERATING ADDITIONAL COMPARISON FIGURES")
print("="*80)

# Load results
print("\nLoading results...")
external_df = pd.read_csv('final_results/external_validation_performance.csv')
comparison_df = pd.read_csv('final_results/internal_vs_external_comparison.csv')

print(f"Loaded {len(external_df)} models")
print("\nModels:")
for model in external_df['Model']:
    print(f"  - {model}")

# ============================================================================
# Figure 1: AUROC Comparison (All Models)
# ============================================================================
print("\n" + "="*80)
print("Figure 1: AUROC Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 7))

models = external_df['Model']
aurocs = external_df['AUROC']

# Create color map based on performance
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
sorted_idx = np.argsort(aurocs)
colors_sorted = [colors[i] for i in sorted_idx]

bars = ax.barh(range(len(models)), aurocs, color=colors_sorted, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (model, auroc) in enumerate(zip(models, aurocs)):
    ax.text(auroc + 0.005, i, f'{auroc:.4f}', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('AUROC (Area Under ROC Curve)', fontweight='bold', fontsize=12)
ax.set_title('External Validation: AUROC Comparison Across All Models', fontweight='bold', fontsize=14)
ax.set_xlim([0.8, 0.88])
ax.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('final_results/additional_figures/auroc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: auroc_comparison.png")

# ============================================================================
# Figure 2: F1 Score Comparison
# ============================================================================
print("\n" + "="*80)
print("Figure 2: F1 Score Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 7))

f1_scores = external_df['F1 Score']
colors_f1 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
sorted_idx_f1 = np.argsort(f1_scores)
colors_f1_sorted = [colors_f1[i] for i in sorted_idx_f1]

bars = ax.barh(range(len(models)), f1_scores, color=colors_f1_sorted, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (model, f1) in enumerate(zip(models, f1_scores)):
    ax.text(f1 + 0.005, i, f'{f1:.4f}', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('F1 Score', fontweight='bold', fontsize=12)
ax.set_title('External Validation: F1 Score Comparison Across All Models', fontweight='bold', fontsize=14)
ax.set_xlim([0.65, 0.75])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('final_results/additional_figures/f1_score_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: f1_score_comparison.png")

# ============================================================================
# Figure 3: Precision Comparison
# ============================================================================
print("\n" + "="*80)
print("Figure 3: Precision Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 7))

precisions = external_df['Precision']
colors_prec = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
sorted_idx_prec = np.argsort(precisions)
colors_prec_sorted = [colors_prec[i] for i in sorted_idx_prec]

bars = ax.barh(range(len(models)), precisions, color=colors_prec_sorted, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (model, prec) in enumerate(zip(models, precisions)):
    ax.text(prec + 0.005, i, f'{prec:.4f}', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('Precision (Positive Predictive Value)', fontweight='bold', fontsize=12)
ax.set_title('External Validation: Precision Comparison Across All Models', fontweight='bold', fontsize=14)
ax.set_xlim([0.58, 0.66])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('final_results/additional_figures/precision_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: precision_comparison.png")

# ============================================================================
# Figure 4: Recall Comparison
# ============================================================================
print("\n" + "="*80)
print("Figure 4: Recall Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 7))

recalls = external_df['Recall']
colors_rec = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
sorted_idx_rec = np.argsort(recalls)
colors_rec_sorted = [colors_rec[i] for i in sorted_idx_rec]

bars = ax.barh(range(len(models)), recalls, color=colors_rec_sorted, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (model, rec) in enumerate(zip(models, recalls)):
    ax.text(rec + 0.005, i, f'{rec:.4f}', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=12)
ax.set_title('External Validation: Recall Comparison Across All Models', fontweight='bold', fontsize=14)
ax.set_xlim([0.75, 0.86])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('final_results/additional_figures/recall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: recall_comparison.png")

# ============================================================================
# Figure 5: Accuracy Comparison
# ============================================================================
print("\n" + "="*80)
print("Figure 5: Accuracy Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 7))

accuracies = external_df['Accuracy']
colors_acc = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
sorted_idx_acc = np.argsort(accuracies)
colors_acc_sorted = [colors_acc[i] for i in sorted_idx_acc]

bars = ax.barh(range(len(models)), accuracies, color=colors_acc_sorted, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (model, acc) in enumerate(zip(models, accuracies)):
    ax.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('Accuracy', fontweight='bold', fontsize=12)
ax.set_title('External Validation: Accuracy Comparison Across All Models', fontweight='bold', fontsize=14)
ax.set_xlim([0.74, 0.79])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('final_results/additional_figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: accuracy_comparison.png")

# ============================================================================
# Figure 6: PR AUC Comparison
# ============================================================================
print("\n" + "="*80)
print("Figure 6: PR AUC Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 7))

pr_aucs = external_df['PR AUC']
colors_pr = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
sorted_idx_pr = np.argsort(pr_aucs)
colors_pr_sorted = [colors_pr[i] for i in sorted_idx_pr]

bars = ax.barh(range(len(models)), pr_aucs, color=colors_pr_sorted, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (model, pr) in enumerate(zip(models, pr_aucs)):
    ax.text(pr + 0.005, i, f'{pr:.4f}', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('PR AUC (Precision-Recall Area Under Curve)', fontweight='bold', fontsize=12)
ax.set_title('External Validation: PR AUC Comparison Across All Models', fontweight='bold', fontsize=14)
ax.set_xlim([0.63, 0.76])
ax.axvline(0.335, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (class prevalence)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('final_results/additional_figures/pr_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: pr_auc_comparison.png")

# ============================================================================
# Figure 7: Comprehensive Multi-Metric Comparison (Grouped Bar Chart)
# ============================================================================
print("\n" + "="*80)
print("Figure 7: Comprehensive Multi-Metric Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(16, 9))

metrics = ['AUROC', 'PR AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(models))
width = 0.14

for i, metric in enumerate(metrics):
    values = external_df[metric]
    offset = (i - len(metrics)/2) * width + width/2
    ax.bar(x + offset, values, width, label=metric, alpha=0.8)

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('External Validation: Comprehensive Performance Comparison (All Metrics)',
             fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0.6, 0.9])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('final_results/additional_figures/comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: comprehensive_metrics_comparison.png")

# ============================================================================
# Figure 8: Performance Heatmap
# ============================================================================
print("\n" + "="*80)
print("Figure 8: Performance Heatmap")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data for heatmap
heatmap_data = external_df[['Model'] + metrics].set_index('Model')

# Normalize each metric to 0-1 range for better visualization
heatmap_normalized = heatmap_data.copy()
for col in heatmap_normalized.columns:
    min_val = heatmap_normalized[col].min()
    max_val = heatmap_normalized[col].max()
    heatmap_normalized[col] = (heatmap_normalized[col] - min_val) / (max_val - min_val)

sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='RdYlGn',
            cbar_kws={'label': 'Score'}, linewidths=0.5, ax=ax,
            vmin=0.6, vmax=0.9)

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Metric', fontweight='bold', fontsize=12)
ax.set_title('External Validation: Performance Heatmap (All Models & Metrics)',
             fontweight='bold', fontsize=14)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('final_results/additional_figures/performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: performance_heatmap.png")

# ============================================================================
# Figure 9: Ranking Summary
# ============================================================================
print("\n" + "="*80)
print("Figure 9: Model Ranking by Metric")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate rank for each metric (1 = best)
rank_data = pd.DataFrame(index=models)
for metric in metrics:
    rank_data[metric] = external_df[metric].rank(ascending=False, method='min').astype(int)

# Plot as heatmap
sns.heatmap(rank_data.T, annot=True, fmt='d', cmap='RdYlGn_r',
            cbar_kws={'label': 'Rank (1=Best)'}, linewidths=0.5, ax=ax,
            vmin=1, vmax=7)

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Metric', fontweight='bold', fontsize=12)
ax.set_title('Model Ranking by Performance Metric (1=Best, 7=Worst)',
             fontweight='bold', fontsize=14)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('final_results/additional_figures/model_ranking_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: model_ranking_heatmap.png")

# ============================================================================
# Figure 10: Radar Chart for Top 3 Models
# ============================================================================
print("\n" + "="*80)
print("Figure 10: Radar Chart (Top 3 Models)")
print("="*80)

# Get top 3 models by AUROC
top3_models = external_df.nlargest(3, 'AUROC')

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='polar')

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for idx, row in top3_models.iterrows():
    values = [row[metric] for metric in metrics]
    values += values[:1]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], markersize=8)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0.6, 0.9)
ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85])
ax.set_yticklabels(['0.65', '0.70', '0.75', '0.80', '0.85'])
ax.grid(True)
ax.set_title('Top 3 Models: Multi-Metric Radar Chart',
             fontweight='bold', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.tight_layout()
plt.savefig('final_results/additional_figures/radar_chart_top3.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: radar_chart_top3.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: ALL FIGURES GENERATED")
print("="*80)

print("\nGenerated figures:")
print("  1. auroc_comparison.png              - AUROC bar chart")
print("  2. f1_score_comparison.png           - F1 Score bar chart")
print("  3. precision_comparison.png          - Precision bar chart")
print("  4. recall_comparison.png             - Recall bar chart")
print("  5. accuracy_comparison.png           - Accuracy bar chart")
print("  6. pr_auc_comparison.png             - PR AUC bar chart")
print("  7. comprehensive_metrics_comparison.png - All metrics grouped bar chart")
print("  8. performance_heatmap.png           - Heatmap of all metrics")
print("  9. model_ranking_heatmap.png         - Ranking heatmap")
print(" 10. radar_chart_top3.png              - Radar chart for top 3 models")

print("\nAll figures saved to: final_results/additional_figures/")
print("\n" + "="*80)
