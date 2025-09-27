# xgb_plot_pr.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os
import glob
import sys
import json

# ---------- Logger (same pattern as your eval) ----------
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'pr_evaluation_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

# ---------- Model paths for each feature type ----------
MODEL_PATHS = {
    'mi': {
        '10000': '/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_10000_run_20250825_160615.joblib',
        '100000': '/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_100000_run_20250825_164742.joblib',
        'full': '/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_full_run_20250825_165926.joblib'
    },
    'fi': {
        '10000': 'saved_models/xgboost_ensemble_fi_features_10000_run_20250828_061856.joblib',
        '100000': 'saved_models/xgboost_ensemble_fi_features_100000_run_20250828_065720.joblib',
        'full': 'saved_models/xgboost_ensemble_fi_features_full_run_20250828_071750.joblib'
    }
}

# ---------- Feature loading helper ----------
def load_features_from_json(json_path):
    """Load selected features from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'selected_names' in data:
        selected_features = data['selected_names']
    elif 'selected_features' in data:
        selected_features = data['selected_features']
    else:
        raise ValueError(f"Could not find feature list in {json_path}")
    
    # Define categorical features (same for all feature sets)
    categorical_features = [
        'ContentRating', 'highest_android_version', 'CurrentVersion',
        'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
    ]
    
    numerical_features = [f for f in selected_features if f not in categorical_features]
    
    return selected_features, categorical_features, numerical_features

# ---------- PR helpers ----------
def _ensemble_proba(models, X):
    """Transform with pipeline's preprocessor, then average classifier probabilities."""
    y_preds_proba = []
    for model in models:
        pre = model.named_steps['preprocessor']
        clf = model.named_steps['classifier']
        X_trans = pre.transform(X)
        clf.set_params(device='cpu')  # ensure CPU prediction
        y_pred = clf.predict_proba(X_trans)[:, 1]
        y_preds_proba.append(y_pred)
    return np.mean(y_preds_proba, axis=0)

def plot_pr_subplot(ax, y_true, y_score, label, size, split_type, is_first_plot=False):
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = float(np.mean(y_true))  # positive prevalence

    # step plot is recommended for PR
    ax.step(r, p, where='post', linewidth=2, label=f'{label} (AP={ap:.4f})')
    
    # Only plot baseline once per subplot
    if is_first_plot:
        ax.hlines(baseline, 0, 1, linestyles='--', label=f'Baseline={baseline:.3f}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        
        # Format size labels nicely
        if size == '10000':
            size_label = '10,000'
        elif size == '100000':
            size_label = '100,000'
        elif size == 'full':
            size_label = 'Full'
        else:
            size_label = str(size)
        
        ax.set_title(f'{size_label} Samples')
        ax.grid(True, alpha=0.3)
    
    ax.legend(loc='lower left')
    return p, r, ap, baseline

def main():
    # ---------- Load data (kept consistent with your current eval) ----------
    print("Loading data…")
    df = pd.read_csv('./content/sample_data/corrected_permacts.csv').dropna()

    # Load data and prepare for evaluation
    X = df
    y = df['status']

    # output dirs
    pr_dump_dir = "/home/umflint.edu/koernerg/pr_dumps"
    os.makedirs(pr_dump_dir, exist_ok=True)
    os.makedirs("pr_plots", exist_ok=True)

    # ---------- Create matrix plot figure ----------
    # Create single figure with two rows (one for each model type)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost Precision-Recall Curves', fontsize=16, fontweight='bold')
    
    # Row labels
    row_labels = ['Mutual Information', 'Feature Importance']
    
    # Store results for JSON dumps
    all_results = {}

    # ---------- Evaluate each model type ----------
    for row_idx, (model_type, json_path) in enumerate([
        ('mi', './top_mi_feature_list/mi_top25_catenc(1)_norm(quantile).json'),
        ('fi', './xgboost_feature_importance_20250827_230123.json')
    ]):
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_type.upper()} MODELS")
        print(f"{'='*60}")
        
        if not os.path.exists(json_path):
            print(f"❌ JSON not found: {json_path}")
            continue
        print(f"Using JSON: {json_path}")
        
        # Store results for JSON dumps
        all_results[model_type] = {}

        # ---------- Evaluate each size ----------
        for i, size in enumerate(['10000', '100000', 'full']):
            print(f"\n== Size: {size} ==")
            
            # Check if model exists for this size and model type
            model_path = MODEL_PATHS[model_type].get(size)
            if not model_path or not os.path.exists(model_path):
                print(f"No {model_type} model available for {size}")
                # Create empty subplot
                axes[row_idx, i].text(0.5, 0.5, f'No {model_type}\nmodel\n{size}', 
                                   ha='center', va='center', transform=axes[row_idx, i].transAxes)
                continue
                
            print(f"Loading {model_type} ensemble: {model_path}")
            models = joblib.load(model_path)

            # Load the correct features for this model type
            selected_features, categorical_features, numerical_features = load_features_from_json(json_path)
            print(f"Selected {len(selected_features)} features: {selected_features[:5]}...")
            
            # Filter data to selected features
            X_selected = X[selected_features]
            
            # Load the correct indices for this sample size
            if size == 'full':
                val_indices = np.load('./standardized_data/val_indices_full.npy')
                test_indices = np.load('./standardized_data/test_indices_full.npy')
            else:
                val_indices = np.load(f'./standardized_data/val_indices_{size}.npy')
                test_indices = np.load(f'./standardized_data/test_indices_{size}.npy')
            
            X_val, y_val = X_selected.loc[val_indices], y.loc[val_indices].to_numpy()
            X_test, y_test = X_selected.loc[test_indices], y.loc[test_indices].to_numpy()

            # VAL (first plot in subplot - sets up axes, baseline, etc.)
            y_val_proba = _ensemble_proba(models, X_val)
            val_p, val_r, val_ap, val_base = plot_pr_subplot(
                axes[row_idx, i], y_val, y_val_proba, 'Validation', size, 'Validation', is_first_plot=True
            )
            print(f"[VAL] AP={val_ap:.4f}, baseline={val_base:.4f}")

            # TEST (second plot in same subplot - just adds the curve)
            y_test_proba = _ensemble_proba(models, X_test)
            test_p, test_r, test_ap, test_base = plot_pr_subplot(
                axes[row_idx, i], y_test, y_test_proba, 'Test', size, 'Test', is_first_plot=False
            )
            print(f"[TEST] AP={test_ap:.4f}, baseline={test_base:.4f}")

            # Set y-axis label based on row (model type)
            if row_idx == 0:  # Mutual Information row
                axes[row_idx, i].set_ylabel('Mutual Information Precision')
            else:  # Feature Importance row
                axes[row_idx, i].set_ylabel('Feature Importance Precision')

            # Store results for JSON dumps
            all_results[model_type][size] = {
                "val": {
                    "precision": [float(x) for x in val_p],
                    "recall": [float(x) for x in val_r],
                    "average_precision": float(val_ap),
                    "baseline_prevalence": float(val_base)
                },
                "test": {
                    "precision": [float(x) for x in test_p],
                    "recall": [float(x) for x in test_r],
                    "average_precision": float(test_ap),
                    "baseline_prevalence": float(test_base)
                }
            }

    # ---------- Save matrix plot ----------
    plt.tight_layout()
    matrix_plot_path = f'pr_plots/xgb_pr_matrix_{timestamp}.png'
    plt.savefig(matrix_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[PLOT] Saved matrix plot to: {matrix_plot_path}")

    # ---------- Save JSON dumps for each model type ----------
    for model_type, size_results in all_results.items():
        for size, results in size_results.items():
            out_json = os.path.join(pr_dump_dir, f"xgboost_{model_type}_pr_{size}.json")
            payload = {
                "model": f"xgboost_{model_type}",
                "size": str(size),
                **results
            }
            with open(out_json, "w") as f:
                json.dump(payload, f)
            print(f"[{model_type.upper()}] Wrote PR dump to {out_json}")

    # restore stdout
    sys.stdout = sys.stdout.terminal
    print(f"PR evaluation log saved to: {log_filename}")

if __name__ == "__main__":
    main()
