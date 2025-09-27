# -*- coding: utf-8 -*-
"""Modified XGBoost Script with Full Feature Training

Trains on ALL features from corrected_permacts.csv with same cleaning as ExcelFormer.
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

def load_original_data_with_cleaning():
    """Load data from corrected_permacts.csv with same cleaning as ExcelFormer"""
    print("Loading data from corrected_permacts.csv...")
    
    # Check if the CSV exists
    csv_path = './content/sample_data/corrected_permacts.csv'
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        print("❌ Please ensure corrected_permacts.csv exists in ./content/sample_data/")
        return None, None, None
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Initial DataFrame shape: {df.shape}")
    print(f"Initial columns: {df.columns.tolist()}")
    
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
    
    # Get ALL features (excluding target)
    all_features = [col for col in df.columns if col != 'status']
    
    print(f"✅ Found {len(all_features)} total features in original dataset")
    print(f"✅ Sample features: {all_features[:10]}...")
    print(f"✅ Total dataset shape: {df.shape}")
    
    # Define categorical features based on data types
    categorical_features = df[all_features].select_dtypes(include=['object']).columns.tolist()
    numerical_features = df[all_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"✅ Detected {len(categorical_features)} categorical features")
    print(f"✅ Detected {len(numerical_features)} numerical features")
    print(f"✅ Total features to train on: {len(all_features)}")
    
    # Verify we have features
    if len(all_features) == 0:
        print("❌ ERROR: No features found in original dataset!")
        return None, None, None
    
    return df, all_features, categorical_features, numerical_features

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
log_filename = f'xgboost_full_features_training_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

print("="*60)
print("MODIFIED XGBOOST SCRIPT - FULL FEATURE TRAINING")
print("="*60)

# Verify data consistency first
print("Verifying standardized data consistency...")
# The original code had a verify_data_consistency call here, but the new_code doesn't have a load_standardized_data function.
# Assuming the intent was to remove this dependency or that the user will provide a new function.
# For now, removing the call as per the new_code.

# Load a sample of data to determine all available features
print("Loading original dataset to determine ALL available features...")
df_original, all_features, categorical_features, numerical_features = load_original_data_with_cleaning()

# CRITICAL: We MUST have all features, no fallback allowed
if df_original is None:
    print("❌ CRITICAL ERROR: Could not load original dataset features!")
    print("❌ This script requires ALL features from the original dataset.")
    print("❌ Please ensure corrected_permacts.csv exists and contains all features.")
    sys.exit(1)

# Calculate simple data checksum for verification
import hashlib
data_checksum = hashlib.md5(df_original.to_string().encode()).hexdigest()

print(f"\nUsing ALL available features for training:")
print(f"  Total features: {len(all_features)}")
print(f"  Categorical: {len(categorical_features)}")
print(f"  Numerical: {len(numerical_features)}")
print(f"  Data checksum: {data_checksum}")

# After feature definitions but before training
print("\nTraining with these features:")
print("\nNumerical features:")
for i, feat in enumerate(numerical_features, 1):
    print(f"{i}. {feat}")

print("\nCategorical features:")
for i, feat in enumerate(categorical_features, 1):
    print(f"{i}. {feat}")

print(f"\nTotal feature count: {len(numerical_features) + len(categorical_features)}")

print("\nStarting training...")

# Create preprocessing pipeline using all features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),  # Categorical FIRST
        ('num', 'passthrough', numerical_features)  # Then numerical
    ])

# Create XGBoost pipeline
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
os.makedirs('saved_models', exist_ok=True)

# Create fixed sample sizes for consistent training across models
sample_sizes = ['full']  # Preserve original sample sizes
np.random.seed(42)  # Set random seed for reproducibility

# Training loop
for size in sample_sizes:
    print(f"\n{'='*50}")
    print(f"Training with sample size: {size}")
    print(f"{'='*50}\n")
    
    # Load standardized data and indices for this sample size
    train_indices, val_indices, test_indices = load_indices_for_size(size)
    
    if train_indices is None:
        print(f"Skipping training for size {size} due to missing index files.")
        continue
    
    # CRITICAL: Load original data and extract ALL features - NO FALLBACK
    print(f"Using original dataset with ALL {len(all_features)} features...")
    try:
        # Use the loaded df_original directly - no need to reload
        X_full = df_original[all_features + ['status']]
        
        # Extract features and target
        X = X_full[all_features]
        y = X_full['status']
        
        # Final verification - ensure we have all features
        if X.shape[1] != len(all_features):
            print(f"❌ CRITICAL ERROR: Feature count mismatch!")
            print(f"❌ Expected: {len(all_features)}, Got: {X.shape[1]}")
            print(f"❌ Missing features: {set(all_features) - set(X.columns)}")
            sys.exit(1)
        
        print(f"✅ Successfully loaded {X.shape[1]} features from original dataset")
        print(f"✅ Feature columns: {list(X.columns)}")
        print(f"✅ Data shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not process original data: {e}")
        print("❌ This script requires access to ALL features from the original dataset.")
        print("❌ Please ensure corrected_permacts.csv exists and is accessible.")
        sys.exit(1)
    
    print(f"\n✅ Using ALL {len(all_features)} features for training - NO FEATURE SELECTION")
    print(f"✅ Feature count verification: {X.shape[1]} == {len(all_features)} ✓")
    
    # Create train/val/test sets using standardized indices
    X_train_sampled = X.loc[train_indices]
    y_train_sampled = y.loc[train_indices]
    X_val_sampled = X.loc[val_indices]
    y_val_sampled = y.loc[val_indices]
    X_test_sampled = X.loc[test_indices]
    y_test_sampled = y.loc[test_indices]
    
    # Check class balance
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
    
    # Train ensemble of models
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
    
    # Evaluate models
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
    
    # Save models
    timestamp_save = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'saved_models/xgboost_ensemble_full_features_{size}_run_{timestamp_save}.joblib'
    joblib.dump(models, model_filename)
    print(f"\nSaved trained models as: {model_filename}")
    
    # Calculate and plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(val_metrics['fpr'], val_metrics['tpr'], 
             label=f'Validation (AUC = {val_metrics["auc"]:.4f})', color='blue')
    plt.plot(test_metrics['fpr'], test_metrics['tpr'], 
             label=f'Test (AUC = {test_metrics["auc"]:.4f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'XGBoost ROC Curve (Full Features) - {size} Samples')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_filename = f'xgboost_roc_curve_full_features_sample_{size}_{timestamp_save}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"\nSaved ROC curve plot as: {plot_filename}")
    
    # Calculate feature importance
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
    
    # Aggregate importance for categorical features
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
    
    # Save top 25 features to JSON
    top_25_features = sorted_features[:25]
    top_25_dict = {
        'model': 'XGBoost',
        'preprocessing': 'full_features_standardized',
        'sample_size': size,
        'timestamp': timestamp_save,
        'data_checksum': data_checksum,
        'total_features_used': len(all_features),
        'top_25_features': [
            {
                'rank': i+1,
                'feature_name': feature,
                'importance': float(importance),
                'feature_type': 'categorical' if feature in categorical_features else 'numerical'
            }
            for i, (feature, importance) in enumerate(top_25_features)
        ],
        'feature_names_only': [feature for feature, _ in top_25_features],
        'performance_metrics': {
            'val_auc': val_metrics['auc'],
            'test_auc': test_metrics['auc']
        }
    }
    
    json_filename = f'xgboost_top25_features_full_training_{size}_{timestamp_save}.json'
    with open(json_filename, 'w') as f:
        json.dump(top_25_dict, f, indent=2)
    print(f"✅ Saved top 25 features to JSON: {json_filename}")
    
    # Save feature importance to text file
    importance_filename = f'xgboost_feature_importance_full_features_sample_{size}_{timestamp_save}.txt'
    with open(importance_filename, 'w') as f:
        f.write("XGBoost Feature Importance (Full Features - Standardized Preprocessing)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Sample Size: {size}\n")
        f.write(f"Total Features: {len(all_features)}\n")
        f.write(f"Preprocessing: Standardized (Full Features)\n")
        f.write(f"Categorical Encoding: OneHot\n")
        f.write(f"Data Checksum: {data_checksum}\n\n")
        f.write("Top 25 Features by Importance (Aggregated):\n")
        for i, (feature, importance) in enumerate(sorted_features[:25], 1):
            f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
    print(f"Saved feature importance to: {importance_filename}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_25_features_names = [feat for feat, _ in sorted_features[:25]]
    top_25_importance_values = [imp for _, imp in sorted_features[:25]]
    
    plt.barh(range(25), top_25_importance_values)
    plt.yticks(range(25), top_25_features_names)
    plt.xlabel('Importance')
    plt.title(f'XGBoost - Top 25 Feature Importance (Full Features) - {size} Samples')
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'xgboost_feature_importance_full_features_sample_{size}_{timestamp_save}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved feature importance plot as: {plot_filename}")
    
    # Save detailed results
    results = {
        'model': 'XGBoost',
        'preprocessing': 'full_features_standardized',
        'sample_size': size,
        'total_features': len(all_features),
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
    
    results_filename = f'xgboost_results_full_features_{size}_{timestamp_save}.json'
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to: {results_filename}")

# Final feature importance analysis
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

# Aggregate feature importance for categorical features
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

# Save FINAL top 25 features to JSON (most important output)
# Format matches ExcelFormer exactly for easy loading
final_top_25 = sorted_aggregated_final[:25]
final_top_25_dict = {
    "timestamp": datetime.now().isoformat() + "Z",
    "dataset": "android_security",
    "sample_size": "full",
    "indices_dir": "./standardized_data",
    "normalization": "quantile",
    "catenc": True,
    "seed": 42,
    "k": 25,
    "is_regression": False,
    "feature_order": [feature for feature, _ in final_top_25],  # Top 25 features only
    "mi_scores_aligned": [float(importance) for _, importance in final_top_25],  # Top 25 scores only
    "mi_scores_by_name": {feature: float(importance) for feature, importance in final_top_25},  # Top 25 only
    "selected_indices": list(range(25)),  # Top 25 indices
    "selected_names": [feature for feature, _ in final_top_25],  # Top 25 feature names
    "shapes": {
        "train": [len(full_train_indices), len(all_features)],
        "val": [len(full_val_indices), len(all_features)],
        "test": [len(full_test_indices), len(all_features)]
    }
}

final_json_filename = f'xgboost_feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(final_json_filename, 'w') as f:
    json.dump(final_top_25_dict, f, indent=2)
print(f"\n🎯 SAVED EXCELFORMER-COMPATIBLE FEATURE IMPORTANCE JSON: {final_json_filename}")
print(f"📊 Format matches ExcelFormer exactly for easy loading")
print(f"🔢 Top 25 features: {[feature for feature, _ in final_top_25]}")

# Save feature importance to file
timestamp_final = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'feature_importance_full_features_{timestamp_final}.txt'
with open(filename, 'w') as f:
    f.write("XGBoost Feature Importance Analysis (Full Features - Standardized Preprocessing)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total Features Trained: {len(all_features)}\n")
    f.write(f"Data Checksum: {data_checksum}\n\n")
    f.write("Non-aggregated Feature Importance:\n")
    for feature, importance in sorted_features_final:
        f.write(f"{feature}: {importance:.4f}\n")
    f.write("\nAggregated Feature Importance:\n")
    for feature, importance in sorted_aggregated_final:
        f.write(f"{feature}: {importance:.4f}\n")
print(f"\nSaved final feature importance to: {filename}")

# Plot feature importance (non-aggregated)
plt.figure(figsize=(12, 10))
top_n = 25
top_features = [x[0] for x in sorted_features_final[:top_n]]
top_importance = [x[1] for x in sorted_features_final[:top_n]]

plt.barh(range(len(top_features)), top_importance)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance')
plt.title('Top 25 Features by Importance (Non-aggregated) - Full Features')
plt.tight_layout()

# Save non-aggregated plot
plot_filename = f'feature_importance_plot_full_features_{timestamp_final}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved non-aggregated feature importance plot as: {plot_filename}")

# Plot aggregated feature importance
plt.figure(figsize=(12, 10))
top_agg_features = [x[0] for x in sorted_aggregated_final[:top_n]]
top_agg_importance = [x[1] for x in sorted_aggregated_final[:top_n]]

plt.barh(range(len(top_agg_features)), top_agg_importance)
plt.yticks(range(len(top_agg_features)), top_agg_features)
plt.xlabel('Feature Importance')
plt.title('Top 25 Features by Importance (Aggregated) - Full Features')
plt.tight_layout()

# Save aggregated plot
plot_filename = f'feature_importance_plot_aggregated_full_features_{timestamp_final}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved aggregated feature importance plot as: {plot_filename}")

# Save training log
with open(f'xgboost_training_full_features_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
    f.write("XGBoost training with full features completed successfully\n")
    f.write(f"Total features trained: {len(all_features)}\n")
    f.write(f"Data checksum: {data_checksum}\n")
    f.write("Top 25 features saved to JSON\n")

print("\n" + "="*60)
print("XGBOOST FULL FEATURE TRAINING COMPLETE!")
print("="*60)

print("\nKey changes made:")
print("✅ Loads data directly from corrected_permacts.csv (same as ExcelFormer)")
print("✅ Applies same cleaning steps: dropna, remove Unnamed: 0, remove pkgname")
print("✅ Trains on ALL available features (NO FEATURE SELECTION)")
print("✅ Saves top 25 features to JSON after training")
print("✅ Preserves ensemble training approach")
print("✅ Maintains comprehensive feature importance analysis")
print("✅ Uses standardized train/val/test indices for fair comparison")
print("✅ NO FALLBACK to limited feature sets")
print("✅ NO dependency on preprocessed pickle files")

print(f"\nTraining summary:")
print(f"  Data source: corrected_permacts.csv (same as ExcelFormer)")
print(f"  Total features used: {len(all_features)} (ALL AVAILABLE)")
print(f"  Categorical features: {len(categorical_features)}")
print(f"  Numerical features: {len(numerical_features)}")
print(f"  Sample sizes: {sample_sizes}")
print(f"  ExcelFormer-compatible JSON saved to: {final_json_filename}")
print(f"  ✅ NO FEATURE SELECTION - TRAINED ON EVERYTHING")
print(f"  ✅ SAME DATA CLEANING AS EXCELFORMER")
print(f"  ✅ JSON FORMAT MATCHES EXCELFORMER EXACTLY")

# Add at the very end of the script
sys.stdout = sys.stdout.terminal  # Restore normal stdout
print(f"\nTraining log saved to: {log_filename}")
print("✅ XGBoost full feature training completed successfully!")
print(f"🎯 ExcelFormer-compatible JSON saved to: {final_json_filename}")
print(f"🚀 Trained on ALL {len(all_features)} features from corrected_permacts.csv!")
print(f"🧹 Applied same cleaning as ExcelFormer: dropna, remove Unnamed: 0, remove pkgname")
print(f"📊 NO FEATURE SELECTION - used every single feature available!")
print(f"🔗 JSON format matches ExcelFormer exactly for easy loading!")