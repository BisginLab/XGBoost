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

# Load the most recent model unless specified
def get_latest_model_path():
    model_files = glob.glob('saved_models/xgboost_ensemble_*.joblib')
    if not model_files:
        raise FileNotFoundError("No saved models found in saved_models directory")
    
    # Group models by size
    size_models = {}
    for model_file in model_files:
        # Extract size from filename
        if 'full' in model_file:
            size = 'full'
        else:
            # Extract number from filename
            size = ''.join(filter(str.isdigit, model_file.split('_')[2]))
        size_models[size] = model_file
    
    # Return the most recent model for each size
    latest_models = {}
    for size, files in size_models.items():
        latest_models[size] = max(files)
    
    return latest_models

# Define the specific model paths for each size
MODEL_PATHS = {
    '10000': '/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_10000_run_20250304_120249.joblib',
    '100000': '/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_100000_run_20250304_120249.joblib',
    'full': '/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_full_run_20250304_120249.joblib'
}

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
    cat_feature_names = cat_encoder.get_feature_names_out()  # Don't pass arguments
    feature_names = numerical_features + list(cat_feature_names)

    # Create a dictionary of feature importances
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    # Aggregate importance for categorical features
    aggregated_importance = {}
    
    # Add numerical features directly
    for feat in numerical_features:
        aggregated_importance[feat] = feature_importance_dict[feat]
    
    # Sum importance for each categorical feature's encoded versions
    for cat_feat in categorical_features:
        cat_importance = sum(
            importance for fname, importance in feature_importance_dict.items() 
            if fname.startswith(cat_feat + '_')
        )
        aggregated_importance[cat_feat] = cat_importance

    # Sort and save/print features
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
            for feat, importance in sorted_features:
                line = f"{feat}: {importance:.4f}"
                print(line)
                f.write(line + '\n')
        print(f"\nSaved feature importance to: {filename}")

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('./content/sample_data/corrected_permacts.csv')
    df = df.dropna(ignore_index=False)
    
    # Define features in order of MI importance from ExcelFormer output (exactly as in training script)
    selected_features = [
        'ContentRating', 'LastUpdated', 'days_since_last_update',
        'highest_android_version', 'privacy_policy_link', 'CurrentVersion',
        'TwoStarRatings', 'isSpamming', 'OneStarRatings', 'FourStarRatings',
        'ThreeStarRatings', 'max_downloads_log', 'lowest_android_version',
        'LenWhatsNew', 'FiveStarRatings', 'STORAGE', 'AndroidVersion',
        'developer_address', 'developer_website', 'LOCATION', 'PHONE',
        'intent', 'DeveloperCategory', 'Genre', 'ReviewsAverage'
    ]

    # Define categorical features (exactly as in training script)
    categorical_features = [
        'ContentRating', 'highest_android_version', 'CurrentVersion',
        'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
    ]

    # Get numerical features (exactly as in training script)
    numerical_features = [f for f in selected_features if f not in categorical_features]
    
    # Prepare X and y
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

        # Plot feature importance
        plot_feature_importance(models, numerical_features, categorical_features, size)

    sys.stdout = sys.stdout.terminal  # Restore normal stdout
    print(f"Evaluation log saved to: {log_filename}")

if __name__ == "__main__":
    main() 