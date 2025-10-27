# -*- coding: utf-8 -*-
"""XGBoost Training Script for Top 25 Features from Feature Importance JSON

Loads the top 25 features from xgboost_feature_importance_20250827_230123.json
and trains XGBoost models on those specific features for sample sizes 10k, 100k, and full.
Follows EXACTLY the same patterns as train_xgboost.py and train_xgb_all_features.py.
"""

import zipfile
import io
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import json
import os

import xgboost as xgb
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib

def encode_and_bind(original_dataframe, feature_to_encode, categories=None):
    """One-hot encode a categorical feature and bind it to the original dataframe"""
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], prefix=feature_to_encode)
    if categories is not None:
        # Ensure all categories are present
        for cat in categories:
            col_name = f"{feature_to_encode}_{cat}"
            if col_name not in dummies.columns:
                dummies[col_name] = 0
        # Reorder columns to match categories
        dummies = dummies.reindex(columns=[f"{feature_to_encode}_{cat}" for cat in categories])
    return dummies

def load_features_from_json(json_path):
    """Load top 25 features from the feature importance JSON file"""
    print(f"Loading features from: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found at: {json_path}")
        print("❌ Please ensure the feature importance JSON file exists")
        return None, None, None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract the selected feature names from the JSON
    selected_features = data['selected_names']
    
    print(f"✅ Loaded {len(selected_features)} features from JSON")
    print(f"✅ Features: {selected_features}")
    
    # Define categorical features based on the selected features
    # These are the features that should be treated as categorical
    categorical_features = [
        'ContentRating', 'highest_android_version', 'CurrentVersion',
        'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
    ]
    
    # Filter to only include categorical features that are in the selected features
    categorical_features = [f for f in categorical_features if f in selected_features]
    
    # Numerical features are the remaining selected features
    numerical_features = [f for f in selected_features if f not in categorical_features]
    
    print(f"✅ Categorical features: {categorical_features}")
    print(f"✅ Numerical features: {numerical_features}")
    
    return selected_features, categorical_features, numerical_features

def load_original_data_with_cleaning():
    """Load data from corrected_permacts.csv with same cleaning as ExcelFormer"""
    print("Loading data from corrected_permacts.csv...")
    
    # Check if the CSV exists
    csv_path = './content/sample_data/corrected_permacts.csv'
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        print("❌ Please ensure corrected_permacts.csv exists in ./content/sample_data/")
        return None
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Initial DataFrame shape: {df.shape}")
    
    # Apply same cleaning as ExcelFormer
    df = df.dropna()
    print(f"Shape after dropping NaNs: {df.shape}")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print(f"Shape after dropping Unnamed: 0: {df.shape}")
    
    # Drop pkgname as in ExcelFormer
    if 'pkgname' in df.columns:
        df = df.drop(['pkgname'], axis=1)
        print(f"Shape after dropping pkgname: {df.shape}")
    
    print(f"✅ Final dataset shape: {df.shape}")
    return df

def load_indices_for_size(size, indices_dir='./standardized_data'):
    """Load train/val/test indices for a specific sample size"""
    size_str = 'full' if size == 'full' else str(size)
    
    train_idx_path = f"{indices_dir}/train_indices_{size_str}.npy"
    val_idx_path = f"{indices_dir}/val_indices_{size_str}.npy"
    test_idx_path = f"{indices_dir}/test_indices_{size_str}.npy"
    
    if not all(os.path.exists(p) for p in [train_idx_path, val_idx_path, test_idx_path]):
        print(f"❌ Missing index files for size {size_str}")
        return None, None, None
    
    train_indices = np.load(train_idx_path)
    val_indices = np.load(val_idx_path)
    test_indices = np.load(test_idx_path)
    
    print(f"Loaded indices for size {size_str}:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices

# Check XGBoost GPU support
print("XGBoost GPU support:", xgb.build_info())

# Check available GPUs
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Add at the start of the script, after imports
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Add after the imports
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'xgboost_fi_features_training_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

print("="*60)
print("XGBOOST TRAINING SCRIPT - TOP 25 FEATURES FROM FI JSON")
print("="*60)

# Load features from the feature importance JSON (ONLY DIFFERENCE)
print("Loading top 25 features from feature importance JSON...")
json_path = './xgboost_feature_importance_20250827_230123.json'
selected_features, categorical_features, numerical_features = load_features_from_json(json_path)

if selected_features is None:
    print("❌ CRITICAL ERROR: Could not load features from JSON!")
    sys.exit(1)

# Load original dataset (SAME AS train_xgb_all_features.py)
print("\nLoading original dataset...")
df_original = load_original_data_with_cleaning()

if df_original is None:
    print("❌ CRITICAL ERROR: Could not load original dataset!")
    sys.exit(1)

# Verify all selected features exist in the dataset
missing_features = [f for f in selected_features if f not in df_original.columns]
if missing_features:
    print(f"❌ CRITICAL ERROR: Missing features in dataset: {missing_features}")
    print(f"❌ Available columns: {df_original.columns.tolist()}")
    sys.exit(1)

print(f"✅ All {len(selected_features)} selected features found in dataset")

# Calculate simple data checksum for verification
import hashlib
data_checksum = hashlib.md5(df_original.to_string().encode()).hexdigest()

print(f"\nTraining on top 25 features from feature importance:")
print(f"  Total features: {len(selected_features)}")
print(f"  Categorical: {len(categorical_features)}")
print(f"  Numerical: {len(numerical_features)}")
print(f"  Data checksum: {data_checksum}")

# After feature definitions but before training (SAME AS OTHER SCRIPTS)
print("\nTraining with these features:")
print("\nNumerical features:")
for i, feat in enumerate(numerical_features, 1):
    print(f"{i}. {feat}")

print("\nCategorical features:")
for i, feat in enumerate(categorical_features, 1):
    print(f"{i}. {feat}")

print("\nFeature order from JSON (CRITICAL for consistency):")
for i, feat in enumerate(selected_features, 1):
    feat_type = "CAT" if feat in categorical_features else "NUM"
    print(f"{i}. {feat} ({feat_type})")

print(f"\nTotal feature count: {len(numerical_features) + len(categorical_features)}")

print("\nStarting training...")

# CRITICAL: Create preprocessing pipeline using canonical order (SAME AS OTHER SCRIPTS)
# This ensures features are in same order as ExcelFormer expects
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),  # Categorical FIRST
        ('num', 'passthrough', numerical_features)  # Then numerical
    ])

# Create XGBoost pipeline (SAME AS OTHER SCRIPTS)
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        enable_categorical=True,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        n_jobs=-1,
        max_bin=256
    ))
])

# Create directory for models if it doesn't exist
os.makedirs('results/xgboost', exist_ok=True)

# Create fixed sample sizes for consistent training across models (SAME AS OTHER SCRIPTS)
sample_sizes = [10000, 100000, 'full']  # Preserve original sample sizes
np.random.seed(42)  # Set random seed for reproducibility

# Training loop - PRESERVE ALL ORIGINAL FUNCTIONALITY (SAME AS OTHER SCRIPTS)
for size in sample_sizes:
    print(f"\n{'='*50}")
    print(f"Training with sample size: {size}")
    print(f"{'='*50}\n")
    
    # Load indices for this sample size (SAME AS train_xgb_all_features.py)
    train_indices, val_indices, test_indices = load_indices_for_size(size)
    
    if train_indices is None:
        print(f"Skipping training for size {size} due to missing index files.")
        continue
    
    # CRITICAL: Load original data and extract selected features - NO FALLBACK
    print(f"Using original dataset with selected {len(selected_features)} features...")
    try:
        # Use the loaded df_original directly - no need to reload
        X_full = df_original[selected_features + ['status']]
        
        # Extract features and target
        X = X_full[selected_features]
        y = X_full['status']
        
        # Final verification - ensure we have all features
        if X.shape[1] != len(selected_features):
            print(f"❌ CRITICAL ERROR: Feature count mismatch!")
            print(f"❌ Expected: {len(selected_features)}, Got: {X.shape[1]}")
            print(f"❌ Missing features: {set(selected_features) - set(X.columns)}")
            sys.exit(1)
        
        print(f"✅ Successfully loaded {X.shape[1]} features from original dataset")
        print(f"✅ Feature columns: {list(X.columns)}")
        print(f"✅ Data shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not process original data: {e}")
        print("❌ This script requires access to selected features from the original dataset.")
        print("❌ Please ensure corrected_permacts.csv exists and is accessible.")
        sys.exit(1)
    
    print(f"\n✅ Using selected {len(selected_features)} features for training")
    print(f"✅ Feature count verification: {X.shape[1]} == {len(selected_features)} ✓")
    
    # Create train/val/test sets using standardized indices (SAME AS OTHER SCRIPTS)
    X_train_sampled = X.loc[train_indices]
    y_train_sampled = y.loc[train_indices]
    X_val_sampled = X.loc[val_indices]
    y_val_sampled = y.loc[val_indices]
    X_test_sampled = X.loc[test_indices]
    y_test_sampled = y.loc[test_indices]
    
    # Check class balance (SAME AS OTHER SCRIPTS)
    print(f"\nClass balance verification:")
    print(f"  Train - Class 0 (benign): {(y_train_sampled == 0).sum()}")
    print(f"  Train - Class 1 (malware): {(y_train_sampled == 1).sum()}")
    print(f"  Train - Ratio: {(y_train_sampled == 1).mean():.3f}")
    print(f"  Val - Class 0 (benign): {(y_val_sampled == 0).sum()}")
    print(f"  Val - Class 1 (malware): {(y_val_sampled == 1).sum()}")
    print(f"  Val - Ratio: {(y_val_sampled == 1).mean():.3f}")
    print(f"  Test - Class 0 (benign): {(y_test_sampled == 0).sum()}")
    print(f"  Test - Class 1 (malware): {(y_test_sampled == 1).sum()}")
    print(f"  Test - Ratio: {(y_test_sampled == 1).mean():.3f}")
    
    # Train ensemble of models (SAME AS OTHER SCRIPTS)
    models = []
    for i in range(11):  # Train 11 models
        print(f"\nTraining model {i+1}/11")
        model = xgb_pipeline.set_params(
            classifier__n_estimators=1000,
            classifier__max_depth=6,
            classifier__learning_rate=0.01,
            classifier__subsample=0.8,
            classifier__colsample_bytree=0.8,
            classifier__min_child_weight=1,
            classifier__gamma=0,
            classifier__random_state=42 + i
        )
        
        model.fit(X_train_sampled, y_train_sampled)
        models.append(model)
    
    # Evaluate models (SAME AS OTHER SCRIPTS)
    def evaluate_models(models, X, y, set_name):
        try:
            # Get predictions from all models
            y_preds_proba = []
            for model in models:
                # Get preprocessor and classifier separately
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                
                # Transform data first
                X_transformed = preprocessor.transform(X)
                
                # Then predict with classifier directly
                classifier.set_params(device='cpu')  # Force CPU prediction
                y_pred = classifier.predict_proba(X_transformed)[:, 1]
                y_preds_proba.append(y_pred)
            
            y_pred_proba = np.mean(y_preds_proba, axis=0)
            
            # Calculate metrics
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc_score = auc(fpr, tpr)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = np.mean(y_pred == y)
            
            print(f'\n{set_name.upper()} SET METRICS:')
            print(f'ROC AUC: {auc_score:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            
            return {'fpr': fpr, 'tpr': tpr, 'auc': auc_score, 'predictions': y_pred_proba}
        except Exception as e:
            print(f"Error in evaluate_models: {str(e)}")
            raise
    
    val_metrics = evaluate_models(models, X_val_sampled, y_val_sampled, "VALIDATION SET")
    test_metrics = evaluate_models(models, X_test_sampled, y_test_sampled, "TEST SET")
    
    # Save models (SAME AS OTHER SCRIPTS)
    timestamp_save = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'results/xgboost/xgboost_ensemble_fi_features_{size}_run_{timestamp_save}.joblib'
    joblib.dump(models, model_filename)
    print(f"\nSaved trained models as: {model_filename}")
    
    # Calculate and plot ROC curve (SAME AS OTHER SCRIPTS)
    plt.figure(figsize=(10, 8))
    plt.plot(val_metrics['fpr'], val_metrics['tpr'], 
             label=f'Validation (AUC = {val_metrics["auc"]:.4f})', color='blue')
    plt.plot(test_metrics['fpr'], test_metrics['tpr'], 
             label=f'Test (AUC = {test_metrics["auc"]:.4f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'XGBoost ROC Curve (FI Features) - {size} Samples')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_filename = f'xgboost_roc_curve_fi_features_sample_{size}_{timestamp_save}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"\nSaved ROC curve plot as: {plot_filename}")
    
    # Calculate feature importance (SAME AS OTHER SCRIPTS)
    print("\nCalculating feature importance from trained models...")
    # Get feature names from preprocessor
    preprocessor = models[0].named_steps['preprocessor']
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = list(cat_feature_names) + numerical_features  # Order: categorical + numerical
    
    feature_importance = np.zeros(len(feature_names))
    for model in models:
        feature_importance += model.named_steps['classifier'].feature_importances_
    feature_importance /= len(models)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Aggregate importance for categorical features (SAME AS OTHER SCRIPTS)
    aggregated_importance = {}
    
    # Sum importance for each categorical feature's encoded versions
    for cat_feat in categorical_features:
        cat_importance = importance_df[importance_df['Feature'].str.startswith(cat_feat + '_')]['Importance'].sum()
        aggregated_importance[cat_feat] = cat_importance
    
    # Add numerical features directly
    for feat in numerical_features:
        aggregated_importance[feat] = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    
    # Sort by importance
    sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Save feature importance to text file (SAME AS OTHER SCRIPTS)
    importance_filename = f'xgboost_feature_importance_fi_features_sample_{size}_{timestamp_save}.txt'
    with open(importance_filename, 'w') as f:
        f.write("XGBoost Feature Importance (FI Features - Original Data)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Sample Size: {size}\n")
        f.write(f"Total Features: {len(selected_features)}\n")
        f.write(f"Preprocessing: Original Data (FI Features)\n")
        f.write(f"Categorical Encoding: OneHot\n")
        f.write(f"Data Checksum: {data_checksum}\n\n")
        f.write("Top 25 Features by Importance (Aggregated):\n")
        for i, (feature, importance) in enumerate(sorted_features[:25], 1):
            f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
    print(f"Saved feature importance to: {importance_filename}")
    
    # Plot feature importance (SAME AS OTHER SCRIPTS)
    plt.figure(figsize=(12, 8))
    top_25_features_names = [feat for feat, _ in sorted_features[:25]]
    top_25_importance_values = [imp for _, imp in sorted_features[:25]]
    
    plt.barh(range(25), top_25_importance_values)
    plt.yticks(range(25), top_25_features_names)
    plt.xlabel('Importance')
    plt.title(f'XGBoost - Top 25 Feature Importance (FI Features) - {size} Samples')
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'xgboost_feature_importance_fi_features_sample_{size}_{timestamp_save}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved feature importance plot as: {plot_filename}")
    
    # Save detailed results (SAME AS OTHER SCRIPTS)
    results = {
        'model': 'XGBoost',
        'preprocessing': 'fi_features_original',
        'sample_size': size,
        'total_features': len(selected_features),
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'val_auc': val_metrics['auc'],
        'test_auc': test_metrics['auc'],
        'categorical_encoding': 'OneHot',
        'data_checksum': data_checksum,
        'timestamp': timestamp_save,
        'class_balance': {
            'train_malware_ratio': float((y_train_sampled == 1).mean()),
            'val_malware_ratio': float((y_val_sampled == 1).mean()),
            'test_malware_ratio': float((y_test_sampled == 1).mean())
        }
    }
    
    results_filename = f'xgboost_results_fi_features_{size}_{timestamp_save}.json'
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to: {results_filename}")

# PRESERVE ALL ORIGINAL FEATURE IMPORTANCE ANALYSIS (SAME AS OTHER SCRIPTS)
print("\n" + "="*60)
print("FINAL FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Load full dataset indices for JSON shapes
print("\nLoading full dataset indices for JSON shapes...")
full_train_indices, full_val_indices, full_test_indices = load_indices_for_size('full')
if full_train_indices is None:
    print("❌ Could not load full dataset indices, using placeholder shapes")
    full_train_indices = np.array([])
    full_val_indices = np.array([])
    full_test_indices = np.array([])

# Calculate feature importance from the last trained models
print("\nCalculating feature importance from trained models...")
first_model = models[0]
xgb_feature_importance = first_model.named_steps['classifier'].feature_importances_
feature_importance = np.zeros_like(xgb_feature_importance)

for model in models:
    # Get feature importance from the XGBoost classifier
    xgb_feature_importance = model.named_steps['classifier'].feature_importances_
    # Accumulate feature importance
    feature_importance += xgb_feature_importance

# Average feature importance across all models
feature_importance /= len(models)

# Get feature names from the encoded data
preprocessor = models[0].named_steps['preprocessor']
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names_final = list(cat_feature_names) + numerical_features

# Create a dictionary of feature importances
feature_importance_dict = dict(zip(feature_names_final, feature_importance))

# Sort features by importance
sorted_features_final = sorted(feature_importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True)

# Print top 25 features with their importance (non-aggregated)
print("\nTop 25 Features by Importance (Non-aggregated):")
for i, (feature, importance) in enumerate(sorted_features_final[:25], 1):
    print(f"{i:2d}. {feature}: {importance:.4f}")

# Aggregate feature importance for categorical features (SAME AS OTHER SCRIPTS)
aggregated_importance_final = {}

# numerical features: copy as-is
for feat in numerical_features:
    aggregated_importance_final[feat] = feature_importance_dict.get(feat, 0.0)

# categorical features: sum all one-hot columns that start with "<parent>_"
for cat_feat in categorical_features:
    prefix = f"{cat_feat}_"
    aggregated_importance_final[cat_feat] = sum(
        imp for fname, imp in feature_importance_dict.items() if fname.startswith(prefix)
    )

# Sort aggregated features by importance
sorted_aggregated_final = sorted(aggregated_importance_final.items(), 
                         key=lambda x: x[1], 
                         reverse=True)

# Print top 25 aggregated features
print("\nTop 25 Features by Importance (Aggregated):")
for i, (feature, importance) in enumerate(sorted_aggregated_final[:25], 1):
    print(f"{i:2d}. {feature}: {importance:.4f}")

# Save feature importance to file (SAME AS OTHER SCRIPTS)
timestamp_final = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'feature_importance_fi_features_{timestamp_final}.txt'
with open(filename, 'w') as f:
    f.write("XGBoost Feature Importance Analysis (FI Features - Original Data)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Data Checksum: {data_checksum}\n")
    f.write(f"Feature Order: {selected_features}\n\n")
    f.write("Non-aggregated Feature Importance:\n")
    for feature, importance in sorted_features_final:
        f.write(f"{feature}: {importance:.4f}\n")
    f.write("\nAggregated Feature Importance:\n")
    for feature, importance in sorted_aggregated_final:
        f.write(f"{feature}: {importance:.4f}\n")
print(f"\nSaved final feature importance to: {filename}")

# Plot feature importance (non-aggregated) - PRESERVE ORIGINAL (SAME AS OTHER SCRIPTS)
plt.figure(figsize=(12, 10))
top_n = 25
top_features = [x[0] for x in sorted_features_final[:top_n]]
top_importance = [x[1] for x in sorted_features_final[:top_n]]

plt.barh(range(len(top_features)), top_importance)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance')
plt.title('Top 25 Features by Importance (Non-aggregated) - FI Features')
plt.tight_layout()

# Save non-aggregated plot
plot_filename = f'feature_importance_plot_fi_features_{timestamp_final}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved non-aggregated feature importance plot as: {plot_filename}")

# Plot aggregated feature importance - PRESERVE ORIGINAL (SAME AS OTHER SCRIPTS)
plt.figure(figsize=(12, 10))
top_agg_features = [x[0] for x in sorted_aggregated_final[:top_n]]
top_agg_importance = [x[1] for x in sorted_aggregated_final[:top_n]]

plt.barh(range(len(top_agg_features)), top_agg_importance)
plt.yticks(range(len(top_agg_features)), top_agg_features)
plt.xlabel('Feature Importance')
plt.title('Top 25 Features by Importance (Aggregated) - FI Features')
plt.tight_layout()

# Save aggregated plot
plot_filename = f'feature_importance_plot_aggregated_fi_features_{timestamp_final}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved aggregated feature importance plot as: {plot_filename}")

# Save training log (SAME AS OTHER SCRIPTS)
with open(f'xgboost_training_fi_features_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
    f.write("XGBoost training with FI features completed successfully\n")
    f.write(f"Data checksum: {data_checksum}\n")
    f.write(f"Feature order: {selected_features}\n")
    f.write("All original functionality preserved\n")

print("\n" + "="*60)
print("XGBOOST FI FEATURES TRAINING COMPLETE!")
print("="*60)

print("\nKey improvements made:")
print("✅ Uses original data loading (same as train_xgb_all_features.py)")
print("✅ Preserves ALL original functionality")
print("✅ Maintains ensemble training approach")
print("✅ Keeps comprehensive feature importance analysis")
print("✅ Uses features from feature importance JSON file")
print("✅ Loads from corrected_permacts.csv with same cleaning")
print("✅ Compatible with ExcelFormer preprocessing")

print(f"\nPreprocessing details:")
print(f"  Data source: corrected_permacts.csv (same as ExcelFormer)")
print(f"  Feature order: From feature importance JSON ({len(selected_features)} features)")
print(f"  Categorical encoding: OneHot")
print(f"  Data checksum: {data_checksum}")
print(f"  Sample sizes: {sample_sizes}")

# Add at the very end of the script
sys.stdout = sys.stdout.terminal  # Restore normal stdout
print(f"\nTraining log saved to: {log_filename}")
print("✅ XGBoost FI features training completed successfully!")
print("✅ All original functionality preserved!")
print("✅ Ready for fair comparison with ExcelFormer!")
print(f"🚀 Trained on top 25 features from feature importance JSON!")
print(f"📊 Trained models for sample sizes: {sample_sizes}")
print(f"🧹 Applied same cleaning as ExcelFormer: dropna, remove Unnamed: 0, remove pkgname")
print(f"🔗 Uses standardized indices for fair comparison!")
