"""
Generate curve-based comparison figures for all 7 models
Similar to ROC curves - showing all models as overlaid curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("GENERATING CURVE-BASED COMPARISON FIGURES")
print("="*80)

# Load data
print("\nLoading preprocessed data and models...")
with open('final_results/preprocessing_objects.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

# Load original data
df = pd.read_csv('data/clinical_genotype_HGB.csv')

exclude_features = [
    'wihsid', 'bsdate', 'bsvisit', 'dob', 'date',
    'lnegdate', 'fposdate', 'frstartd', 'frstaidd', 'frstdthd',
    'undetectable', 'HIV', 'r',
    'vload', 'logvl', 'vla', 'cd8a',
    'status', 'n', 'N', 'visit'
]

feature_cols = [col for col in df.columns if col not in exclude_features]
X = df[feature_cols].copy()
y = df['undetectable'].copy()

mask = y.notna()
X = X[mask]
y = y[mask].astype(int)

# Split to get external validation set
from sklearn.model_selection import train_test_split
X_internal, X_external, y_internal, y_external = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Preprocess external set
from sklearn.preprocessing import LabelEncoder
label_encoders = preprocessing['label_encoders']
categorical_features = X_external.select_dtypes(include=['object']).columns.tolist()

X_external_encoded = X_external.copy()
for col in categorical_features:
    X_external_encoded[col] = label_encoders[col].transform(
        X_external_encoded[col].astype(str).replace('nan', 'MISSING')
    )

X_external_imputed = preprocessing['imputer'].transform(X_external_encoded)
X_external_scaled = preprocessing['scaler'].transform(X_external_imputed)

# Load models
print("Loading trained models...")
models = {}
model_files = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'SVM': 'svm_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'K-Nearest Neighbors': 'knn_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl'
}

for name, filename in model_files.items():
    with open(f'final_results/{filename}', 'rb') as f:
        models[name] = pickle.load(f)

# Load DNN
from tensorflow import keras
models['Deep Neural Network'] = keras.models.load_model('final_results/dnn_model.keras')

# Create output directory
output_dir = Path('final_results')
output_dir.mkdir(parents=True, exist_ok=True)

# Define colors for models
colors = plt.cm.tab10(np.linspace(0, 0.9, len(models)))
color_map = dict(zip(models.keys(), colors))

print("\n" + "="*80)
print("1. GENERATING ROC CURVES (ALL MODELS)")
print("="*80)

plt.figure(figsize=(12, 10))

for name, model in models.items():
    print(f"Processing {name}...")

    if name == 'Deep Neural Network':
        y_pred_proba = model.predict(X_external_scaled, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_external_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_external, y_pred_proba)
    auroc = roc_auc_score(y_external, y_pred_proba)

    plt.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUROC = {auroc:.3f})',
             color=color_map[name], alpha=0.85)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.6)
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=13)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=13)
plt.title('ROC Curves: All 7 Models on External Validation Set',
          fontweight='bold', fontsize=15, pad=15)
plt.legend(loc='lower right', fontsize=11, framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(output_dir / 'roc_curves_external.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'roc_curves_external.png'}")
plt.close()

print("\n" + "="*80)
print("2. GENERATING PRECISION-RECALL CURVES (ALL MODELS)")
print("="*80)

plt.figure(figsize=(12, 10))

for name, model in models.items():
    print(f"Processing {name}...")

    if name == 'Deep Neural Network':
        y_pred_proba = model.predict(X_external_scaled, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_external_scaled)[:, 1]

    precision, recall, _ = precision_recall_curve(y_external, y_pred_proba)
    pr_auc = average_precision_score(y_external, y_pred_proba)

    plt.plot(recall, precision, linewidth=2.5,
             label=f'{name} (PR AUC = {pr_auc:.3f})',
             color=color_map[name], alpha=0.85)

# Baseline (proportion of positive class)
baseline = y_external.sum() / len(y_external)
plt.plot([0, 1], [baseline, baseline], 'k--', linewidth=2,
         label=f'Baseline (Random) = {baseline:.3f}', alpha=0.6)

plt.xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=13)
plt.ylabel('Precision (PPV)', fontweight='bold', fontsize=13)
plt.title('Precision-Recall Curves: All 7 Models on External Validation Set',
          fontweight='bold', fontsize=15, pad=15)
plt.legend(loc='best', fontsize=11, framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(output_dir / 'pr_curves_external.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'pr_curves_external.png'}")
plt.close()

print("\n" + "="*80)
print("3. GENERATING THRESHOLD ANALYSIS CURVES")
print("="*80)

# For each model, show how precision, recall, and F1 vary with threshold
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    print(f"Processing {name}...")

    if name == 'Deep Neural Network':
        y_pred_proba = model.predict(X_external_scaled, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_external_scaled)[:, 1]

    precision, recall, thresholds_pr = precision_recall_curve(y_external, y_pred_proba)

    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Use thresholds from precision-recall curve
    # Note: precision_recall_curve returns one more value than thresholds
    thresholds = thresholds_pr
    precision_plot = precision[:-1]
    recall_plot = recall[:-1]
    f1_plot = f1_scores[:-1]

    ax = axes[idx]
    ax.plot(thresholds, precision_plot, linewidth=2.5, label='Precision',
            color='#2E86AB', alpha=0.85)
    ax.plot(thresholds, recall_plot, linewidth=2.5, label='Recall',
            color='#A23B72', alpha=0.85)
    ax.plot(thresholds, f1_plot, linewidth=2.5, label='F1 Score',
            color='#F18F01', alpha=0.85)

    # Mark optimal threshold (max F1)
    optimal_idx = np.argmax(f1_plot)
    optimal_threshold = thresholds[optimal_idx]
    ax.axvline(x=optimal_threshold, color='green', linestyle='--',
              linewidth=2, alpha=0.5, label=f'Optimal = {optimal_threshold:.2f}')

    ax.set_xlabel('Threshold', fontweight='bold', fontsize=10)
    ax.set_ylabel('Score', fontweight='bold', fontsize=10)
    ax.set_title(name, fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

# Remove empty subplot
fig.delaxes(axes[-1])

plt.suptitle('Threshold Analysis: Precision, Recall, F1 vs Classification Threshold\n(External Validation)',
             fontweight='bold', fontsize=15, y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'threshold_analysis_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'threshold_analysis_curves.png'}")
plt.close()

print("\n" + "="*80)
print("4. GENERATING METRIC COMPARISON CURVES")
print("="*80)

# Read performance data
external_df = pd.read_csv('final_results/external_validation_performance.csv')

# Plot each metric as a curve across models
metrics_to_plot = ['AUROC', 'PR AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_colors = {
    'AUROC': '#2E86AB',
    'PR AUC': '#A23B72',
    'Accuracy': '#F18F01',
    'Precision': '#C73E1D',
    'Recall': '#6A994E',
    'F1 Score': '#BC4B51'
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]

    models_list = external_df['Model'].tolist()
    values = external_df[metric].values
    x_positions = np.arange(len(models_list))

    # Plot as smooth curve
    from scipy.interpolate import make_interp_spline

    # Create smooth curve
    x_smooth = np.linspace(x_positions.min(), x_positions.max(), 300)
    spl = make_interp_spline(x_positions, values, k=2)  # k=2 for quadratic
    values_smooth = spl(x_smooth)

    # Plot smooth curve
    ax.plot(x_smooth, values_smooth, linewidth=3, color=metric_colors[metric],
            alpha=0.6, label=f'{metric} (smooth)')

    # Plot actual points
    ax.scatter(x_positions, values, s=150, color=metric_colors[metric],
              alpha=1.0, zorder=5, edgecolors='black', linewidths=2)

    # Add value labels
    for i, (x, y) in enumerate(zip(x_positions, values)):
        ax.annotate(f'{y:.3f}',
                   xy=(x, y),
                   xytext=(0, 12),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=metric_colors[metric], alpha=0.8))

    # Styling
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Across All Models', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models_list, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([max(0.5, values.min() - 0.05), min(1.0, values.max() + 0.05)])

    # Add best performance line
    best_val = values.max()
    ax.axhline(y=best_val, color='green', linestyle='--', alpha=0.3, linewidth=2)

plt.suptitle('Performance Metrics Comparison: All 7 Models\n(External Validation)',
             fontweight='bold', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'metrics_curves_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'metrics_curves_comparison.png'}")
plt.close()

print("\n" + "="*80)
print("5. GENERATING COMPREHENSIVE MULTI-METRIC CURVE")
print("="*80)

plt.figure(figsize=(14, 10))

models_list = external_df['Model'].tolist()
x_positions = np.arange(len(models_list))

for metric in metrics_to_plot:
    values = external_df[metric].values

    plt.plot(x_positions, values, marker='o', linewidth=2.5, markersize=10,
             label=metric, alpha=0.85, color=metric_colors[metric])

plt.xlabel('Model', fontweight='bold', fontsize=13)
plt.ylabel('Score', fontweight='bold', fontsize=13)
plt.title('Comprehensive Performance Comparison: All Metrics Across All Models\n(External Validation)',
          fontweight='bold', fontsize=15, pad=15)
plt.xticks(x_positions, models_list, rotation=45, ha='right', fontsize=11)
plt.legend(loc='best', fontsize=12, framealpha=0.95, ncol=2)
plt.grid(True, alpha=0.3)
plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.savefig(output_dir / 'comprehensive_metrics_curve.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'comprehensive_metrics_curve.png'}")
plt.close()

print("\n" + "="*80)
print("✓ ALL CURVE-BASED VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\nLocation: {output_dir}/")
print("\nGenerated files:")
print("  1. roc_curves_external.png - ROC curves for all models")
print("  2. pr_curves_external.png - Precision-Recall curves for all models")
print("  3. threshold_analysis_curves.png - Threshold analysis for each model")
print("  4. metrics_curves_comparison.png - Individual metric curves (6 subplots)")
print("  5. comprehensive_metrics_curve.png - All metrics on one plot")
print("="*80)
