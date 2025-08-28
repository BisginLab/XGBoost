#!/usr/bin/env python3
"""
Quick script to regenerate ROC curve plots with legend in bottom right
without retraining models.
"""

import matplotlib.pyplot as plt
import joblib
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
from master_preprocessing import load_standardized_data

# Sample sizes to process
sample_sizes = ['10000', '100000', 'full']

# Timestamps from your training run
timestamps = {
    '10000': '20250719_214955',
    '100000': '20250719_221934', 
    'full': '20250719_225326'
}

for size in sample_sizes:
    timestamp = timestamps[size]
    
    print(f"\nProcessing {size} sample size...")
    
    # Load the saved models
    model_file = f'saved_models/xgboost_ensemble_standardized_{size}_run_{timestamp}.joblib'
    models = joblib.load(model_file)
    print(f"Loaded models from: {model_file}")
    
    # Load the standardized data 
    df_clean, train_indices, val_indices, test_indices, metadata = load_standardized_data(
        sample_size=size, data_dir='./standardized_data'
    )
    
    # Get canonical feature order
    canonical_feature_order = metadata['canonical_feature_order']
    
    # Extract features and target
    X = df_clean[canonical_feature_order]
    y = df_clean['status']
    
    # Create validation and test sets
    X_val = X.loc[val_indices]
    y_val = y.loc[val_indices]
    X_test = X.loc[test_indices] 
    y_test = y.loc[test_indices]
    
    # Get predictions from ensemble
    def get_ensemble_predictions(models, X, y):
        y_preds_proba = []
        for model in models:
            # Get preprocessor and classifier
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            
            # Transform data and predict
            X_transformed = preprocessor.transform(X)
            classifier.set_params(device='cpu')  # Force CPU prediction
            y_pred = classifier.predict_proba(X_transformed)[:, 1]
            y_preds_proba.append(y_pred)
        
        y_pred_proba = np.mean(y_preds_proba, axis=0)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score
    
    # Get validation and test metrics
    val_fpr, val_tpr, val_auc = get_ensemble_predictions(models, X_val, y_val)
    test_fpr, test_tpr, test_auc = get_ensemble_predictions(models, X_test, y_test)
    
    # Format size with commas for title
    if size.isdigit():
        size_formatted = f"{int(size):,}"
    else:
        size_formatted = size.title()  # Capitalize 'full' -> 'Full'
    
    # Create the plot with legend in bottom right
    plt.figure(figsize=(10, 8))
    plt.plot(val_fpr, val_tpr, 
             label=f'Validation (AUC = {val_auc:.4f})', color='blue')
    plt.plot(test_fpr, test_tpr, 
             label=f'Test (AUC = {test_auc:.4f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'XGBoost ROC Curve (Standardized) - {size_formatted} Samples')
    plt.legend(loc='lower right')  # Changed to bottom right!
    plt.grid(True)
    
    # Save the new plot
    new_plot_filename = f'xgboost_roc_curve_standardized_sample_{size}_{timestamp}_fixed_legend.png'
    plt.savefig(new_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved new ROC curve plot: {new_plot_filename}")
    print(f"   Validation AUC: {val_auc:.4f}")
    print(f"   Test AUC: {test_auc:.4f}")

print("\n🎉 All ROC curve plots regenerated with legend in bottom right!")
print("\nNew files created:")
for size in sample_sizes:
    timestamp = timestamps[size]
    print(f"  - xgboost_roc_curve_standardized_sample_{size}_{timestamp}_fixed_legend.png") 