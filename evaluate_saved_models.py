import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os
import glob

# Load the most recent model unless specified
def get_latest_model_path():
    model_files = glob.glob('saved_models/xgboost_ensemble_*.joblib')
    if not model_files:
        raise FileNotFoundError("No saved models found in saved_models directory")
    return max(model_files)  # Gets most recent by filename

def evaluate_models(models, X, y, set_name):
    # Get predictions from all models
    y_preds_proba = [model.predict_proba(X)[:, 1] for model in models]
    y_pred_proba = np.mean(y_preds_proba, axis=0)
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    # Calculate Accuracy
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = np.mean(y_pred == y)
    
    print(f'\n{set_name.upper()} SET METRICS:')
    print(f'ROC AUC: {auc_score:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    
    return fpr, tpr, auc_score

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('./content/sample_data/corrected_permacts.csv')
    df = df.dropna(ignore_index=False)
    
    # Define features (same as training script)
    selected_features = [
        'Unnamed: 0', 'ContentRating', 'LastUpdated', 'days_since_last_update',
        'highest_android_version', 'pkgname', 'privacy_policy_link', 'CurrentVersion',
        'TwoStarRatings', 'isSpamming', 'OneStarRatings', 'FourStarRatings',
        'ThreeStarRatings', 'max_downloads_log', 'lowest_android_version',
        'LenWhatsNew', 'FiveStarRatings', 'STORAGE', 'AndroidVersion',
        'developer_address', 'developer_website', 'LOCATION', 'PHONE',
        'intent', 'DeveloperCategory', 'Genre', 'ReviewsAverage'
    ]
    
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
    
    # Load saved models
    model_path = get_latest_model_path()
    print(f"\nLoading models from: {model_path}")
    models = joblib.load(model_path)
    
    # Evaluate and plot
    print("\nEvaluating saved models...")
    val_fpr, val_tpr, val_auc = evaluate_models(models, X_val, y_val, "validation")
    test_fpr, test_tpr, test_auc = evaluate_models(models, X_test, y_test, "test")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(val_fpr, val_tpr, color='blue', lw=2, label=f'Validation (AUC = {val_auc:.4f})')
    plt.plot(test_fpr, test_tpr, color='red', lw=2, label=f'Test (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Evaluation of Saved Models)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'evaluation_roc_curve_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved evaluation plot as: {plot_filename}")

if __name__ == "__main__":
    main() 