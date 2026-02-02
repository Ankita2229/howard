"""
Notebook 16: Complete Model Comparison with KNN and Decision Tree
=================================================================
This notebook adds K-Nearest Neighbors (KNN) and Decision Tree models
to the existing 5 models from notebook 15, creating a comprehensive
comparison of 7 machine learning algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import shap
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("NOTEBOOK 16: COMPLETE MODEL COMPARISON (7 MODELS)")
print("="*80)
print("\nLibraries loaded successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"XGBoost version: {xgb.__version__}")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING DATA")
print("="*80)

df = pd.read_csv('data/clinical_genotype_HGB.csv')

print(f"\nOriginal dataset shape: {df.shape}")
print(f"\nTarget variable distribution:")
print(df['undetectable'].value_counts())
print(f"\nMissing targets: {df['undetectable'].isna().sum()} ({df['undetectable'].isna().sum()/len(df)*100:.1f}%)")

# Exclude data leakage features
exclude_features = [
    'wihsid', 'bsdate', 'bsvisit', 'dob', 'date',
    'lnegdate', 'fposdate', 'frstartd', 'frstaidd', 'frstdthd',
    'undetectable', 'HIV', 'r',
    'vload', 'logvl', 'vla', 'cd8a',  # DATA LEAKAGE FEATURES
    'status', 'n', 'N', 'visit'
]

feature_cols = [col for col in df.columns if col not in exclude_features]

X = df[feature_cols].copy()
y = df['undetectable'].copy()

# Remove missing targets
mask = y.notna()
X = X[mask]
y = y[mask].astype(int)

print(f"\nFeature set: {len(feature_cols)} features")
print(f"Valid samples: {X.shape[0]}")
print(f"\nClass distribution:")
print(f"  Not suppressed (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  Suppressed (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  Imbalance ratio: {(y==0).sum()/(y==1).sum():.2f}:1")

# ============================================================================
# 2. CREATE EXTERNAL VALIDATION SET (20%)
# ============================================================================
print("\n" + "="*80)
print("2. CREATING EXTERNAL VALIDATION SET (20%)")
print("="*80)

X_internal, X_external, y_internal, y_external = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\nInternal (80% - for training/tuning):")
print(f"  Total samples: {len(X_internal)}")
print(f"  Class 0 (not suppressed): {(y_internal==0).sum()} ({(y_internal==0).sum()/len(y_internal)*100:.1f}%)")
print(f"  Class 1 (suppressed): {(y_internal==1).sum()} ({(y_internal==1).sum()/len(y_internal)*100:.1f}%)")

print(f"\nExternal Validation (20% - HELD OUT):")
print(f"  Total samples: {len(X_external)}")
print(f"  Class 0 (not suppressed): {(y_external==0).sum()} ({(y_external==0).sum()/len(y_external)*100:.1f}%)")
print(f"  Class 1 (suppressed): {(y_external==1).sum()} ({(y_external==1).sum()/len(y_external)*100:.1f}%)")

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("3. PREPROCESSING")
print("="*80)

# Encode categorical features
label_encoders = {}
categorical_features = X_internal.select_dtypes(include=['object']).columns.tolist()

X_internal_encoded = X_internal.copy()
for col in categorical_features:
    le = LabelEncoder()
    X_internal_encoded[col] = le.fit_transform(X_internal_encoded[col].astype(str).replace('nan', 'MISSING'))
    label_encoders[col] = le

X_external_encoded = X_external.copy()
for col in categorical_features:
    X_external_encoded[col] = label_encoders[col].transform(X_external_encoded[col].astype(str).replace('nan', 'MISSING'))

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_internal_imputed = imputer.fit_transform(X_internal_encoded)
X_external_imputed = imputer.transform(X_external_encoded)

# Scale features
scaler = StandardScaler()
X_internal_scaled = scaler.fit_transform(X_internal_imputed)
X_external_scaled = scaler.transform(X_external_imputed)

print(f"\nInternal data preprocessed: {X_internal_scaled.shape}")
print(f"External data preprocessed: {X_external_scaled.shape}")

# ============================================================================
# 4. APPLYING SMOTE OVERSAMPLING
# ============================================================================
print("\n" + "="*80)
print("4. APPLYING SMOTE OVERSAMPLING")
print("="*80)

print(f"\nBefore SMOTE:")
print(f"  Class 0: {(y_internal==0).sum()}")
print(f"  Class 1: {(y_internal==1).sum()}")
print(f"  Ratio: {(y_internal==0).sum()/(y_internal==1).sum():.2f}:1")

smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')
X_internal_balanced, y_internal_balanced = smote.fit_resample(X_internal_scaled, y_internal)

print(f"\nAfter SMOTE:")
print(f"  Class 0: {(y_internal_balanced==0).sum()}")
print(f"  Class 1: {(y_internal_balanced==1).sum()}")
print(f"  Ratio: {(y_internal_balanced==0).sum()/(y_internal_balanced==1).sum():.2f}:1")
print(f"\nBalanced internal data shape: {X_internal_balanced.shape}")

# Split balanced internal data
X_train, X_val, y_train, y_val = train_test_split(
    X_internal_balanced, y_internal_balanced,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_internal_balanced
)

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# ============================================================================
# 5. TRAINING MODELS
# ============================================================================
print("\n" + "="*80)
print("5. TRAINING MODELS")
print("="*80)

# 5.1 Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_val_pred = lr_model.predict_proba(X_val)[:, 1]
lr_val_auroc = roc_auc_score(y_val, lr_val_pred)
print(f"Validation AUROC: {lr_val_auroc:.4f}")

# 5.2 SVM
print("\nTraining SVM...")
svm_model = SVC(C=0.1, kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
svm_model.fit(X_train, y_train)
svm_val_pred = svm_model.predict_proba(X_val)[:, 1]
svm_val_auroc = roc_auc_score(y_val, svm_val_pred)
print(f"Validation AUROC: {svm_val_auroc:.4f}")

# 5.3 Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=10,
    min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict_proba(X_val)[:, 1]
rf_val_auroc = roc_auc_score(y_val, rf_val_pred)
print(f"Validation AUROC: {rf_val_auroc:.4f}")

# 5.4 XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
    eval_metric='logloss', n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_val_auroc = roc_auc_score(y_val, xgb_val_pred)
print(f"Validation AUROC: {xgb_val_auroc:.4f}")

# 5.5 Deep Neural Network
print("\nTraining Deep Neural Network...")
n_features = X_train.shape[1]
dnn_model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
dnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['AUC', 'accuracy']
)

early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max', verbose=0)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6, mode='max', verbose=0)

history = dnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)
dnn_val_pred = dnn_model.predict(X_val, verbose=0).flatten()
dnn_val_auroc = roc_auc_score(y_val, dnn_val_pred)
print(f"Validation AUROC: {dnn_val_auroc:.4f}")

# 5.6 K-Nearest Neighbors (NEW)
print("\nTraining K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(
    n_neighbors=11,  # Optimized value
    weights='distance',  # Weight by inverse distance
    metric='minkowski',
    p=2,  # Euclidean distance
    n_jobs=-1
)
knn_model.fit(X_train, y_train)
knn_val_pred = knn_model.predict_proba(X_val)[:, 1]
knn_val_auroc = roc_auc_score(y_val, knn_val_pred)
print(f"Validation AUROC: {knn_val_auroc:.4f}")

# 5.7 Decision Tree (NEW)
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=10,  # Limit depth to prevent overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)
dt_model.fit(X_train, y_train)
dt_val_pred = dt_model.predict_proba(X_val)[:, 1]
dt_val_auroc = roc_auc_score(y_val, dt_val_pred)
print(f"Validation AUROC: {dt_val_auroc:.4f}")

# ============================================================================
# 6. INTERNAL VALIDATION (5-FOLD CROSS-VALIDATION)
# ============================================================================
print("\n" + "="*80)
print("6. INTERNAL VALIDATION (5-FOLD CROSS-VALIDATION)")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models_cv = {
    'Logistic Regression': lr_model,
    'SVM': svm_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'K-Nearest Neighbors': knn_model,
    'Decision Tree': dt_model
}

cv_results = []

for name, model in models_cv.items():
    print(f"\nCross-validating {name}...")
    scores = cross_val_score(model, X_internal_balanced, y_internal_balanced,
                            cv=cv, scoring='roc_auc', n_jobs=-1)

    cv_results.append({
        'Model': name,
        'Mean CV AUROC': scores.mean(),
        'Std CV AUROC': scores.std(),
        'Min CV AUROC': scores.min(),
        'Max CV AUROC': scores.max(),
        '95% CI Lower': scores.mean() - 1.96*scores.std(),
        '95% CI Upper': scores.mean() + 1.96*scores.std()
    })

    print(f"  AUROC: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  95% CI: [{scores.mean() - 1.96*scores.std():.4f}, {scores.mean() + 1.96*scores.std():.4f}]")

# Add DNN with validation AUROC
cv_results.append({
    'Model': 'Deep Neural Network',
    'Mean CV AUROC': dnn_val_auroc,
    'Std CV AUROC': np.nan,
    'Min CV AUROC': np.nan,
    'Max CV AUROC': np.nan,
    '95% CI Lower': np.nan,
    '95% CI Upper': np.nan
})

print(f"\nValidating Deep Neural Network...")
print(f"  AUROC: {dnn_val_auroc:.4f}")

cv_df = pd.DataFrame(cv_results)
print("\n" + "="*80)
print("INTERNAL CV RESULTS")
print("="*80)
print("\n" + cv_df.to_string(index=False))

# ============================================================================
# 7. EXTERNAL VALIDATION (HELD-OUT TEST SET)
# ============================================================================
print("\n" + "="*80)
print("7. EXTERNAL VALIDATION (HELD-OUT TEST SET)")
print("="*80)

print(f"\nExternal set: {X_external_scaled.shape[0]} samples (NEVER seen during training)")

external_results = []

models_all = {
    'Logistic Regression': lr_model,
    'SVM': svm_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Deep Neural Network': dnn_model,
    'K-Nearest Neighbors': knn_model,
    'Decision Tree': dt_model
}

for name, model in models_all.items():
    print(f"\nEvaluating {name}...")

    if name == 'Deep Neural Network':
        y_pred_proba = model.predict(X_external_scaled, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_external_scaled)[:, 1]

    y_pred = (y_pred_proba >= 0.5).astype(int)

    auroc = roc_auc_score(y_external, y_pred_proba)
    pr_auc = average_precision_score(y_external, y_pred_proba)
    accuracy = accuracy_score(y_external, y_pred)
    precision = precision_score(y_external, y_pred)
    recall = recall_score(y_external, y_pred)
    f1 = f1_score(y_external, y_pred)

    external_results.append({
        'Model': name,
        'AUROC': auroc,
        'PR AUC': pr_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    print(f"  AUROC: {auroc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

external_df = pd.DataFrame(external_results)
print("\n" + "="*80)
print("EXTERNAL VALIDATION RESULTS")
print("="*80)
print("\n" + external_df.to_string(index=False))

# ============================================================================
# 8. INTERNAL vs EXTERNAL PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("8. INTERNAL vs EXTERNAL PERFORMANCE COMPARISON")
print("="*80)

comparison = []

for i, row in external_df.iterrows():
    model_name = row['Model']

    if model_name in cv_df['Model'].values:
        internal_auroc = cv_df[cv_df['Model'] == model_name]['Mean CV AUROC'].values[0]
    else:
        internal_auroc = dnn_val_auroc

    external_auroc = row['AUROC']
    gap = internal_auroc - external_auroc

    if gap < 0:
        risk = 'None (External > Internal)'
    elif gap < 0.02:
        risk = 'Low'
    elif gap < 0.05:
        risk = 'Medium'
    else:
        risk = 'High'

    comparison.append({
        'Model': model_name,
        'Internal AUROC': internal_auroc,
        'External AUROC': external_auroc,
        'Gap': gap,
        'Overfitting Risk': risk
    })

comparison_df = pd.DataFrame(comparison)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("9. SAVING RESULTS")
print("="*80)

os.makedirs('final_results', exist_ok=True)

cv_df.to_csv('final_results/internal_cv_results.csv', index=False)
external_df.to_csv('final_results/external_validation_performance.csv', index=False)
comparison_df.to_csv('final_results/internal_vs_external_comparison.csv', index=False)

# Save models
models_to_save = {
    'logistic_regression': lr_model,
    'svm': svm_model,
    'random_forest': rf_model,
    'xgboost': xgb_model,
    'knn': knn_model,
    'decision_tree': dt_model
}

for name, model in models_to_save.items():
    filename = f'final_results/{name}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

dnn_model.save('final_results/dnn_model.keras')

preprocessing_objects = {
    'imputer': imputer,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': X.columns.tolist()
}

with open('final_results/preprocessing_objects.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)

print("\nAll results saved to 'final_results/' directory!")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("10. GENERATING VISUALIZATIONS")
print("="*80)

# 10.1 Internal vs External Comparison
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['Internal AUROC'], width,
               label='Internal (CV)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, comparison_df['External AUROC'], width,
               label='External (Held-Out)', color='darkorange', alpha=0.8)

for i, (_, row) in enumerate(comparison_df.iterrows()):
    gap = row['Gap']
    color = 'green' if gap < 0.02 else ('orange' if gap < 0.05 else 'red')
    ax.annotate(f'Gap: {gap:.3f}',
               xy=(i, max(row['Internal AUROC'], row['External AUROC']) + 0.01),
               ha='center', fontsize=9, color=color, fontweight='bold')

ax.set_xlabel('Model', fontweight='bold')
ax.set_ylabel('AUROC', fontweight='bold')
ax.set_title('Internal vs External Performance: All 7 Models', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'], rotation=30, ha='right')
ax.legend()
ax.set_ylim([0.75, 1.0])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('final_results/internal_vs_external_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.2 ROC Curves
plt.figure(figsize=(11, 9))

for name, model in models_all.items():
    if name == 'Deep Neural Network':
        y_pred_proba = model.predict(X_external_scaled, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_external_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_external, y_pred_proba)
    auroc = roc_auc_score(y_external, y_pred_proba)

    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUROC = {auroc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
plt.title('ROC Curves: All 7 Models on External Validation Set', fontweight='bold', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_results/roc_curves_external.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.3 Confusion Matrices
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, (name, model) in enumerate(models_all.items()):
    if name == 'Deep Neural Network':
        y_pred_proba = model.predict(X_external_scaled, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_external_scaled)[:, 1]

    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_external, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Suppressed', 'Suppressed'],
                yticklabels=['Not Suppressed', 'Suppressed'],
                cbar=True, square=True)
    axes[idx].set_xlabel('Predicted', fontweight='bold')
    axes[idx].set_ylabel('Actual', fontweight='bold')
    axes[idx].set_title(f'{name}', fontweight='bold', fontsize=11)

# Remove empty subplots
for idx in range(len(models_all), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Confusion Matrices: All 7 Models on External Validation Set',
             fontweight='bold', fontsize=15, y=0.995)
plt.tight_layout()
plt.savefig('final_results/confusion_matrices_external.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualizations saved!")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EXECUTION COMPLETE!")
print("="*80)

print("\nAll results saved to 'final_results/' directory")
print("\nKey files:")
print("  - internal_cv_results.csv")
print("  - external_validation_performance.csv")
print("  - internal_vs_external_comparison.csv")
print("  - internal_vs_external_comparison.png")
print("  - roc_curves_external.png")
print("  - confusion_matrices_external.png")
print("  - All trained models (*.pkl and *.keras)")

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

best_external = external_df.sort_values('AUROC', ascending=False).iloc[0]
print(f"\nBest External AUROC: {best_external['Model']}")
print(f"  AUROC: {best_external['AUROC']:.4f}")
print(f"  F1 Score: {best_external['F1 Score']:.4f}")

best_f1 = external_df.sort_values('F1 Score', ascending=False).iloc[0]
print(f"\nBest F1 Score: {best_f1['Model']}")
print(f"  AUROC: {best_f1['AUROC']:.4f}")
print(f"  F1 Score: {best_f1['F1 Score']:.4f}")

most_overfit = comparison_df.sort_values('Gap', ascending=False).iloc[0]
print(f"\nMost Overfitting: {most_overfit['Model']}")
print(f"  Gap: {most_overfit['Gap']:.4f} ({most_overfit['Overfitting Risk']} risk)")

best_generalization = comparison_df.sort_values('Gap', ascending=True).iloc[0]
print(f"\nBest Generalization: {best_generalization['Model']}")
print(f"  Gap: {best_generalization['Gap']:.4f} ({best_generalization['Overfitting Risk']} risk)")

print("\n" + "="*80)
