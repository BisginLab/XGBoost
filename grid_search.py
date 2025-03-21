import sys
import pkg_resources
import warnings

# Required packages and versions
REQUIRED_PACKAGES = {
    'xgboost': '2.0.3',
    'torch': '2.0.0',
    'scikit-learn': '1.0.0',
    'pandas': '1.5.0',
    'numpy': '1.23.0',
    'psutil': '5.9.0',
    'joblib': '1.1.0'
}

def check_packages():
    """Check if required packages are installed with correct versions"""
    missing = []
    outdated = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            installed = pkg_resources.get_distribution(package)
            if pkg_resources.parse_version(installed.version) < pkg_resources.parse_version(min_version):
                outdated.append(f"{package} (installed: {installed.version}, required: {min_version})")
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    if missing or outdated:
        print("\nPackage Requirements Issues:")
        if missing:
            print("\nMissing packages:")
            print("\n".join(f"  - {pkg}" for pkg in missing))
            print("\nInstall with:")
            print(f"pip install {' '.join(missing)}")
        
        if outdated:
            print("\nOutdated packages:")
            print("\n".join(f"  - {pkg}" for pkg in outdated))
            print("\nUpdate with:")
            print("pip install --upgrade " + " ".join(pkg.split()[0] for pkg in outdated))
        
        return False
    return True

# Check packages before importing
if not check_packages():
    print("\nPlease install/update the required packages and try again.")
    sys.exit(1)

# Now import the rest of the packages
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import os
import joblib
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import psutil
import traceback

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Print package versions for debugging
print("\nPackage versions:")
print(f"Python: {sys.version}")
print(f"XGBoost: {xgb.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"scikit-learn: {pkg_resources.get_distribution('scikit-learn').version}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")

# Check CUDA configuration
print("\nCUDA Configuration:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"XGBoost GPU support: {xgb.build_info()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Logger class for saving output
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

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'grid_search_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

def load_data():
    print("\nLoading data...")
    df = pd.read_csv('./content/sample_data/corrected_permacts.csv')
    df = df.dropna(ignore_index=False)
    print(f"Shape after dropping NaNs: {df.shape}")
    return df

def get_features():
    # Define features based on MI ranking
    selected_features = [
        'ContentRating', 'Genre', 'CurrentVersion', 'AndroidVersion', 
        'DeveloperCategory', 'lowest_android_version', 'highest_android_version',
        'privacy_policy_link', 'developer_website', 'days_since_last_update',
        'isSpamming', 'max_downloads_log', 'LenWhatsNew', 'PHONE',
        'OneStarRatings', 'developer_address', 'FourStarRatings', 'intent',
        'ReviewsAverage', 'STORAGE', 'LastUpdated', 'TwoStarRatings',
        'LOCATION', 'FiveStarRatings', 'ThreeStarRatings'
    ]

    categorical_features = [
        'ContentRating', 'highest_android_version', 'CurrentVersion',
        'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
    ]

    numerical_features = [f for f in selected_features if f not in categorical_features]
    
    return selected_features, categorical_features, numerical_features

def validate_data(df, features):
    """Validate data before processing"""
    print("\nDEBUG - Data Validation:")
    
    # Check for NaN values
    nan_cols = df[features].isna().sum()
    print("\nColumns with NaN values:")
    print(nan_cols[nan_cols > 0])
    
    # Check data types
    print("\nData types:")
    print(df[features].dtypes)
    
    # Check unique values in categorical columns
    categorical_features = [
        'ContentRating', 'highest_android_version', 'CurrentVersion',
        'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
    ]
    print("\nUnique values in categorical columns:")
    for col in categorical_features:
        n_unique = df[col].nunique()
        print(f"{col}: {n_unique} unique values")

def encode_and_bind(original_dataframe, feature_to_encode, encoder=None):
    """Encode categorical feature and bind to dataframe, matching train_xgboost.py approach"""
    if encoder is None:
        # If no encoder provided, create new one (training mode)
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        return dummies, pd.get_dummies(original_dataframe[[feature_to_encode]], prefix=feature_to_encode)
    else:
        # Use provided encoder (validation mode)
        dummy_cols = [col for col in encoder.columns if col.startswith(feature_to_encode + '_')]
        current_dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], prefix=feature_to_encode)
        # Add missing columns
        for col in dummy_cols:
            if col not in current_dummies:
                current_dummies[col] = 0
        # Reorder columns to match training data
        current_dummies = current_dummies[dummy_cols]
        return encoder[dummy_cols], current_dummies

def create_pipeline(device='cuda:0'):
    selected_features, categorical_features, numerical_features = get_features()
    
    print("\nDEBUG - Pipeline Configuration:")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Current GPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            
            # Test CUDA
            torch.cuda.empty_cache()
            test_tensor = torch.cuda.FloatTensor(1)
            print("CUDA test successful")
            device = 'cuda:0'
        except Exception as e:
            print(f"CUDA initialization failed: {str(e)}")
            print("Falling back to CPU")
            device = 'cpu'
    else:
        device = 'cpu'
        print("GPU not available, using CPU")
    
    print(f"\nUsing device: {device}")
    
    try:
        # Configure XGBoost classifier only - no preprocessing pipeline needed
        classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            tree_method='gpu_hist' if device == 'cuda:0' else 'hist',
            predictor='gpu_predictor' if device == 'cuda:0' else 'cpu_predictor',
            n_jobs=-1,
            max_bin=256,
            gpu_id=0 if device == 'cuda:0' else None,
            verbosity=2
        )
        
        return classifier
        
    except Exception as e:
        print(f"\nClassifier creation error:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

def grid_search(X_train, y_train, X_val, y_val):
    print("\nDEBUG - Starting Grid Search Setup")
    
    try:
        # Memory check before starting
        print("\nInitial Memory Status:")
        print(f"RAM Usage: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
        print(f"RAM Available: {psutil.virtual_memory().available / 1e9:.2f} GB")
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        categorical_features = [
            'ContentRating', 'highest_android_version', 'CurrentVersion',
            'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
        ]
        
        # Initialize encoded dataframes with numerical features
        numerical_features = [col for col in X_train.columns if col not in categorical_features]
        X_train_encoded = X_train[numerical_features].copy()
        X_val_encoded = X_val[numerical_features].copy()
        
        # Encode each categorical feature
        for feature in categorical_features:
            print(f"Encoding {feature}...")
            # Get encodings from training data
            train_dummies, train_encoded = encode_and_bind(X_train, feature)
            # Apply same encoding to validation data
            _, val_encoded = encode_and_bind(X_val, feature, train_encoded)
            
            # Add encoded features to dataframes
            X_train_encoded = pd.concat([X_train_encoded, train_encoded], axis=1)
            X_val_encoded = pd.concat([X_val_encoded, val_encoded], axis=1)
        
        print(f"\nEncoded shapes:")
        print(f"X_train: {X_train_encoded.shape}")
        print(f"X_val: {X_val_encoded.shape}")
        
        # Verify column alignment
        print("\nVerifying column alignment...")
        train_cols = set(X_train_encoded.columns)
        val_cols = set(X_val_encoded.columns)
        if train_cols != val_cols:
            missing_in_val = train_cols - val_cols
            missing_in_train = val_cols - train_cols
            if missing_in_val:
                print(f"Columns missing in validation set: {missing_in_val}")
            if missing_in_train:
                print(f"Columns missing in training set: {missing_in_train}")
            raise ValueError("Train and validation sets have different columns")
        
        print("Column alignment verified")
        
        # Define parameter grid
        estimators_list = [256, 512]
        depths_list = [2, 3]
        learning_rates = [0.01, 0.1]
        subsample_rates = [0.8]
        colsample_rates = [0.8]
        min_child_weights = [1]
        gamma_values = [0.1]
        
        print("\nParameter grid:")
        print(f"n_estimators: {estimators_list}")
        print(f"max_depth: {depths_list}")
        print(f"learning_rate: {learning_rates}")
        
        best_score = 0
        best_params = None
        best_model = None
        
        # Manual grid search
        total_combinations = (len(estimators_list) * len(depths_list) * 
                            len(learning_rates) * len(subsample_rates) * 
                            len(colsample_rates) * len(min_child_weights) * 
                            len(gamma_values))
        
        print(f"\nTotal parameter combinations to try: {total_combinations}")
        current_combination = 0
        
        for n_est in estimators_list:
            for max_d in depths_list:
                for lr in learning_rates:
                    for subsample in subsample_rates:
                        for colsample in colsample_rates:
                            for min_child in min_child_weights:
                                for gamma in gamma_values:
                                    current_combination += 1
                                    print(f"\nTrying combination {current_combination}/{total_combinations}")
                                    print(f"Parameters: n_estimators={n_est}, max_depth={max_d}, learning_rate={lr}")
                                    
                                    # Create and configure classifier
                                    classifier = xgb.XGBClassifier(
                                        n_estimators=n_est,
                                        max_depth=max_d,
                                        learning_rate=lr,
                                        subsample=subsample,
                                        colsample_bytree=colsample,
                                        min_child_weight=min_child,
                                        gamma=gamma,
                                        objective='binary:logistic',
                                        eval_metric='auc',
                                        use_label_encoder=False,
                                        tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
                                        predictor='gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
                                        n_jobs=-1,
                                        max_bin=256,
                                        gpu_id=0 if torch.cuda.is_available() else None,
                                        verbosity=2
                                    )
                                    
                                    # Train the model
                                    classifier.fit(
                                        X_train_encoded, 
                                        y_train,
                                        eval_set=[(X_val_encoded, y_val)],
                                        early_stopping_rounds=10,
                                        verbose=True
                                    )
                                    
                                    # Get validation score
                                    val_pred = classifier.predict_proba(X_val_encoded)[:, 1]
                                    val_score = roc_auc_score(y_val, val_pred)
                                    
                                    print(f"Validation AUC: {val_score:.4f}")
                                    
                                    # Update best model if necessary
                                    if val_score > best_score:
                                        best_score = val_score
                                        best_params = {
                                            'n_estimators': n_est,
                                            'max_depth': max_d,
                                            'learning_rate': lr,
                                            'subsample': subsample,
                                            'colsample_bytree': colsample,
                                            'min_child_weight': min_child,
                                            'gamma': gamma
                                        }
                                        best_model = classifier
                                        
                                        print("\nNew best model found!")
                                        print(f"Best validation AUC: {best_score:.4f}")
                                        print("Best parameters:", best_params)
                                    
                                    # Clear GPU memory
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
        
        # Create a GridSearchCV-like results object
        class GridSearchResults:
            def __init__(self, best_estimator_, best_params_, best_score_):
                self.best_estimator_ = best_estimator_
                self.best_params_ = best_params_
                self.best_score_ = best_score_
        
        return GridSearchResults(best_model, best_params, best_score)
        
    except Exception as e:
        print("\nDEBUG - Detailed error information:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        
        # Memory diagnostics
        print("\nMemory status at error:")
        print(f"RAM Usage: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
        print(f"RAM Available: {psutil.virtual_memory().available / 1e9:.2f} GB")
        
        if torch.cuda.is_available():
            print("\nGPU Memory status:")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            
            print("\nAttempting to clear GPU memory...")
            torch.cuda.empty_cache()
        
        raise

def save_results(grid_search, timestamp):
    # Create results directory
    results_dir = 'grid_search_results'
    os.makedirs(results_dir, exist_ok=True)

    # Save best model
    best_model_path = f'{results_dir}/best_model_{timestamp}.joblib'
    joblib.dump(grid_search.best_estimator_, best_model_path)
    print(f"\nSaved best model to: {best_model_path}")

    # Save grid search results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    
    results_path = f'{results_dir}/grid_search_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved grid search results to: {results_path}")

    # Print best parameters and score
    print("\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nBest validation score: {grid_search.best_score_:.4f}")

def main():
    try:
        # Load data
        df = load_data()
        selected_features, _, _ = get_features()

        # Prepare X and y
        X = df[selected_features]
        y = df['status']

        # Load train/val/test splits
        train_indices = np.load('./content/sample_data/train_indices.npy')
        val_indices = np.load('./content/sample_data/val_indices.npy')

        # Split data using loaded indices
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_val = X.loc[val_indices]
        y_val = y.loc[val_indices]

        # Create results directory
        results_dir = 'grid_search_results'
        os.makedirs(results_dir, exist_ok=True)

        # Perform grid search
        grid_result = grid_search(X_train, y_train, X_val, y_val)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_results(grid_result, timestamp)

    except Exception as e:
        print("\nFatal error in main:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 