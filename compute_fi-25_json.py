#!/usr/bin/env python3
"""
Compute FI-25 JSON (Feature Importance top 25) from XGBoost training.

This script extracts the top 25 features by feature importance from XGBoost
models and saves them to a JSON file.

Behavior:
- First checks for existing trained XGBoost model in /workspace/results/xgboost/
- If found, loads the model (fast)
- If not found, trains new ensemble of 11 XGBoost models on ALL features (slow)

Fixed configuration:
- sample_size: full
- seed: 42
- ensemble of 11 XGBoost models
- aggregates feature importance across categorical encodings

Output: data/feature_regimes/fi-25_features_{timestamp}.json

Usage:
    python compute_fi-25_json.py
    
Note: To generate a fresh model, first run:
    python train_xgboost.py --features all
"""

import json
import os
from datetime import datetime
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import master preprocessing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
from master_preprocessing import load_standardized_data

def find_existing_model(model_dir='/workspace/results/xgboost', feature_mode='all', sample_size='full'):
    """Find the most recent saved XGBoost model for the given configuration"""
    pattern = os.path.join(model_dir, f'xgboost_ensemble_{feature_mode}_{sample_size}_run_*.joblib')
    models = glob.glob(pattern)
    
    if not models:
        return None
    
    # Sort by modification time, most recent first
    models.sort(key=os.path.getmtime, reverse=True)
    return models[0]

def load_all_features_from_csv():
    """Load data from corrected_permacts.csv and get ALL available features"""
    print("Loading ALL features from corrected_permacts.csv...")
    
    csv_path = '/workspace/data/raw/corrected_permacts.csv'
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        return None, None, None
    
    df = pd.read_csv(csv_path)
    print(f"Initial DataFrame shape: {df.shape}")
    
    # Apply same cleaning
    df = df.dropna()
    print(f"Shape after dropping NaNs: {df.shape}")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    if 'pkgname' in df.columns:
        df = df.drop(['pkgname'], axis=1)
    
    # Get ALL features (excluding target)
    all_features = [col for col in df.columns if col != 'status']
    
    print(f"✅ Found {len(all_features)} total features")
    
    # Define categorical features based on data types
    categorical_features = df[all_features].select_dtypes(include=['object']).columns.tolist()
    numerical_features = df[all_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"✅ Categorical features: {len(categorical_features)}")
    print(f"✅ Numerical features: {len(numerical_features)}")
    
    return all_features, categorical_features, numerical_features

def main():
    # Fixed configuration
    sample_size = 'full'
    seed = 42
    n_models = 11
    output_dir = '/workspace/data/feature_regimes'
    
    print("="*60)
    print("COMPUTING FI-25 JSON WITH FIXED CONFIGURATION")
    print("="*60)
    print(f"Sample size: {sample_size}")
    print(f"Random seed: {seed}")
    print(f"Number of models in ensemble: {n_models}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load ALL features
    print("Step 1: Loading feature definitions...")
    selected_features, categorical_features, numerical_features = load_all_features_from_csv()
    if selected_features is None:
        print("❌ Failed to load features!")
        sys.exit(1)
    
    print(f"\nFeature summary:")
    print(f"  Total features: {len(selected_features)}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Numerical: {len(numerical_features)}")
    
    # Load standardized data
    print(f"\nStep 2: Loading standardized data (sample_size={sample_size})...")
    df_clean, train_indices, val_indices, test_indices, metadata = load_standardized_data(
        sample_size=sample_size,
        data_dir='/workspace/data/splits'
    )
    
    print(f"  Train set size: {len(train_indices)}")
    print(f"  Validation set size: {len(val_indices)}")
    print(f"  Test set size: {len(test_indices)}")
    
    # Extract features
    X = df_clean[selected_features]
    y = df_clean['status']
    
    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    
    print(f"  Training data shape: {X_train.shape}")
    
    # Step 3: Check for existing model
    print("\nStep 3: Checking for existing trained models...")
    existing_model_path = find_existing_model(
        model_dir='/workspace/results/xgboost',
        feature_mode='all',
        sample_size='full'
    )
    
    if existing_model_path:
        print(f"✅ Found existing model: {existing_model_path}")
        print("  Loading saved models...")
        try:
            models = joblib.load(existing_model_path)
            print(f"  Successfully loaded {len(models)} models from disk")
            print("  Skipping training step")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("  Will train new models instead")
            existing_model_path = None
    
    if not existing_model_path:
        print("⚠️  No existing model found or failed to load")
        print("  Will train new models...")
        
        # Create preprocessing pipeline
        print("\nStep 4: Creating preprocessing pipeline...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features)
            ]
        )
        
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
        
        # Train ensemble of models
        print(f"\nStep 5: Training ensemble of {n_models} XGBoost models...")
        models = []
        for i in range(n_models):
            print(f"  Training model {i+1}/{n_models}...", end=" ")
            model = xgb_pipeline.set_params(
                classifier__n_estimators=1000,
                classifier__max_depth=6,
                classifier__learning_rate=0.01,
                classifier__subsample=0.8,
                classifier__colsample_bytree=0.8,
                classifier__min_child_weight=1,
                classifier__gamma=0,
                classifier__random_state=seed + i
            )
            
            model.fit(X_train, y_train)
            models.append(model)
            print("✅")
    
    # Calculate feature importance
    print("\nStep 6: Calculating feature importance...")
    
    # Get feature names from preprocessor
    preprocessor = models[0].named_steps['preprocessor']
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = list(cat_feature_names) + numerical_features
    
    # Average feature importance across all models
    feature_importance = np.zeros(len(feature_names))
    for model in models:
        feature_importance += model.named_steps['classifier'].feature_importances_
    feature_importance /= len(models)
    
    # Create feature importance dictionary
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    
    # Aggregate importance for categorical features
    print("  Aggregating categorical feature importance...")
    aggregated_importance = {}
    
    # Sum importance for each categorical feature's encoded versions
    for cat_feat in categorical_features:
        cat_importance = sum(
            imp for feat, imp in feature_importance_dict.items() 
            if feat.startswith(cat_feat + '_')
        )
        aggregated_importance[cat_feat] = cat_importance
    
    # Add numerical features directly
    for feat in numerical_features:
        aggregated_importance[feat] = feature_importance_dict[feat]
    
    # Sort by importance
    sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 25
    top_25 = sorted_features[:25]
    top_25_names = [feat for feat, _ in top_25]
    top_25_importance = {feat: float(imp) for feat, imp in top_25}
    
    print(f"\nTop 25 features by importance:")
    for i, (feature, importance) in enumerate(top_25, 1):
        print(f"  {i:2d}. {feature}: {importance:.4f}")
    
    # Create JSON payload (similar structure to MI-25 JSON)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": "xgboost_feature_importance",
        "sample_size": sample_size,
        "indices_dir": "/workspace/data/splits",
        "seed": seed,
        "n_models": n_models,
        "k": 25,
        "xgboost_params": {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0
        },
        "feature_order": selected_features,  # all features used for training
        "feature_importance_aggregated": aggregated_importance,  # all features with importance
        "selected_names": top_25_names,  # top 25 feature names
        "selected_importance": top_25_importance,  # top 25 with their importance scores
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "total_features": len(selected_features),
        "encoding": "OneHot"
    }
    
    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    json_name = f"fi-25_features_{timestamp}.json"
    out_path = os.path.join(output_dir, json_name)
    
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n✅ Saved FI-25 JSON to: {out_path}")
    
    # Also save a plain-text list
    txt_path = out_path.replace(".json", "_names.txt")
    with open(txt_path, "w") as f:
        f.write(f"Top 25 Features by XGBoost Feature Importance\n")
        f.write("="*60 + "\n\n")
        for i, name in enumerate(top_25_names, 1):
            importance = top_25_importance[name]
            f.write(f"{i:2d}. {name} (Importance: {importance:.4f})\n")
    
    print(f"✅ Saved feature list to: {txt_path}")
    
    print("\n" + "="*60)
    print("FI-25 JSON GENERATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

