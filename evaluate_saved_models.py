import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os
import glob
import sys
import warnings
import json

# Define Logger class first
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

# Then use it
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'evaluation_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

# Use hardcoded paths for the FI features models
MODEL_PATHS = {
    '10000': 'saved_models/xgboost_ensemble_fi_features_10000_run_20250828_061856.joblib',
    '100000': 'saved_models/xgboost_ensemble_fi_features_100000_run_20250828_065720.joblib',
    'full': 'saved_models/xgboost_ensemble_fi_features_full_run_20250828_071750.joblib'
}

print("Using FI features models:")
for size, path in MODEL_PATHS.items():
    print(f"  {size}: {path}")

def evaluate_models(models, X, y, set_name, size):
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
        
        print(f'\n{set_name.upper()} SET METRICS (Size: {size}):')
        print(f'ROC AUC: {auc_score:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        
        return fpr, tpr, auc_score
    except Exception as e:
        print(f"Error in evaluate_models: {str(e)}")
        print("Model structure:", model.named_steps.keys())
        raise

def plot_feature_importance(models, numerical_features, categorical_features, size, save_to_file=True):
    print(f"\nCalculating feature importance for size: {size}")
    # Get the correct shape from the first model
    first_model = models[0]
    xgb_feature_importance = first_model.named_steps['classifier'].feature_importances_
    feature_importance = np.zeros_like(xgb_feature_importance)

    for model in models:
        # Get feature importance from the XGBoost classifier
        xgb_feature_importance = model.named_steps['classifier'].feature_importances_
        feature_importance += xgb_feature_importance

    # Average feature importance across all models
    feature_importance /= len(models)

    # Map feature importance back to original features
    preprocessor = models[0].named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out()
    feature_names = numerical_features + list(cat_feature_names)

    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # Aggregate importance for categorical features
    aggregated_importance = {}
    
    # Add numerical features directly
    for feat in numerical_features:
        aggregated_importance[feat] = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    
    # Sum importance for each categorical feature's encoded versions
    for cat_feat in categorical_features:
        cat_importance = importance_df[importance_df['Feature'].str.startswith(cat_feat + '_')]['Importance'].sum()
        aggregated_importance[cat_feat] = cat_importance

    # Sort features by importance
    sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 25 feature names
    top_25_features = [feat[0] for feat in sorted_features[:25]]
    
    # Save to npy file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    np_filename = f'top_25_xgboost_features_{size}_{timestamp}.npy'
    np.save(np_filename, np.array(top_25_features))
    print(f"\nSaved top 25 features to: {np_filename}")
    
    if save_to_file:
        filename = f'feature_importance_{size}_{timestamp}.txt'
        with open(filename, 'w') as f:
            f.write("Top 25 Features by Importance (Aggregated):\n")
            for i, (feat, importance) in enumerate(sorted_features, 1):
                line = f"{i}. {feat}: {importance:.4f}"
                print(line)
                f.write(line + '\n')
        print(f"\nSaved feature importance to: {filename}")

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

def main():
    # Load features from the same JSON file used for training
    print("Loading features from feature importance JSON...")
    json_path = './xgboost_feature_importance_20250827_230123.json'
    selected_features, categorical_features, numerical_features = load_features_from_json(json_path)
    
    if selected_features is None:
        print("❌ CRITICAL ERROR: Could not load features from JSON!")
        sys.exit(1)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('./content/sample_data/corrected_permacts.csv')
    df = df.dropna(ignore_index=False)
    
    # Verify all selected features exist in the dataset
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        print(f"❌ CRITICAL ERROR: Missing features in dataset: {missing_features}")
        print(f"❌ Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    print(f"✅ All {len(selected_features)} selected features found in dataset")
    
    # Prepare X and y using the features from JSON
    X = df[selected_features]
    y = df['status']
    
    # Load indices
    train_indices = np.load('./content/sample_data/train_indices.npy')
    val_indices = np.load('./content/sample_data/val_indices.npy')
    test_indices = np.load('./content/sample_data/test_indices.npy')
    
    # Split data
    X_val = X.loc[val_indices]
    y_val = y.loc[val_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    
    # Evaluate models for each size
    for size, model_path in MODEL_PATHS.items():
        print(f"\nEvaluating models for size: {size}")
        print(f"Loading models from: {model_path}")
        models = joblib.load(model_path)
        
        # Evaluate and plot
        val_fpr, val_tpr, val_auc = evaluate_models(models, X_val, y_val, "validation", size)
        test_fpr, test_tpr, test_auc = evaluate_models(models, X_test, y_test, "test", size)
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(val_fpr, val_tpr, color='blue', lw=2, label=f'Validation (AUC = {val_auc:.4f})')
        plt.plot(test_fpr, test_tpr, color='red', lw=2, label=f'Test (AUC = {test_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'XGBoost ROC Curves - {size} Samples')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'evaluation_roc_curve_{size}_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved evaluation plot as: {plot_filename}")

        # Save ROC metrics to JSON
        dump_dir = "/home/umflint.edu/koernerg/roc_dumps"
        os.makedirs(dump_dir, exist_ok=True)
        out_json = os.path.join(dump_dir, f"xgboost_fi_features_{size}.json")
        payload = {
            "model": "xgboost_fi_features",
            "size": str(size),
            "val": {
                "fpr": [float(x) for x in val_fpr],
                "tpr": [float(x) for x in val_tpr],
                "auc": float(val_auc),
            },
            "test": {
                "fpr": [float(x) for x in test_fpr],
                "tpr": [float(x) for x in test_tpr],
                "auc": float(test_auc),
            },
        }
        with open(out_json, "w") as f:
            json.dump(payload, f)
        print(f"[XGB] Wrote ROC dump to {out_json}")

        # Plot feature importance
        plot_feature_importance(models, numerical_features, categorical_features, size)

    sys.stdout = sys.stdout.terminal  # Restore normal stdout
    print(f"Evaluation log saved to: {log_filename}")

if __name__ == "__main__":
    main() 