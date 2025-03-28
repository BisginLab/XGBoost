# -*- coding: utf-8 -*-
"""Bisgin_Paper_PyFile.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hGYdWMVD5setW3SNQVujIiFAYZQk8fN6
"""

import zipfile
import io
import pandas as pd
import numpy as np
import sys
from datetime import datetime

import xgboost as xgb
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import joblib

# Check XGBoost GPU support
print("XGBoost GPU support:", xgb.build_info())

# Check available GPUs
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# TO DO
# double check user vs developer centric features from paper
# double check validation and training/testing split
# do evaluation for BOTH user and developer centered
# Explaining misclassifications with llm
# remove unnnamed index
# compare earlier dataset with corrected dataset
# make colab copy to folderr
# push progress to github repo
# do google slides for excelformer, add to google drive

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
import sys
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'training_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

# Read the CSV file
print("Loading data...")
df = pd.read_csv('./content/sample_data/corrected_permacts.csv')
print(f"Initial DataFrame shape: {df.shape}")

# Drop NaNs first, but preserve the original index
df = df.dropna(ignore_index=False)
print(f"Shape after dropping NaNs: {df.shape}")

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

# Define categorical features
categorical_features = [
    'ContentRating', 'highest_android_version', 'CurrentVersion',
    'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
]

# Get numerical features
numerical_features = [f for f in selected_features if f not in categorical_features]

# After feature definitions but before training
print("\nTraining with these features:")
print("\nNumerical features:")
for i, feat in enumerate(numerical_features, 1):
    print(f"{i}. {feat}")

print("\nCategorical features:")
for i, feat in enumerate(categorical_features, 1):
    print(f"{i}. {feat}")

print("\nTotal feature count:", len(numerical_features) + len(categorical_features))
print("\nFeatures to be used in training:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i}. {feat}")

print("\nStarting training...")

# Prepare X and y
X = df[selected_features]
y = df['status']

print("\nFeature counts:")
print(f"Total selected features: {len(selected_features)}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Load the indices that were created after dropping NaNs
train_indices = np.load('./content/sample_data/train_indices.npy')
val_indices = np.load('./content/sample_data/val_indices.npy')
test_indices = np.load('./content/sample_data/test_indices.npy')

# After loading indices but before splitting
print("\nIndices info:")
print(f"Number of train indices: {len(train_indices)}")
print(f"Number of val indices: {len(val_indices)}")
print(f"Number of test indices: {len(test_indices)}")
print(f"Min/Max train indices: {train_indices.min()}, {train_indices.max()}")
print(f"Min/Max val indices: {val_indices.min()}, {val_indices.max()}")
print(f"Min/Max test indices: {test_indices.min()}, {test_indices.max()}")

# Check DataFrame index
print("\nDataFrame info:")
print(f"DataFrame index range: {df.index.min()}, {df.index.max()}")
print(f"DataFrame index is continuous: {df.index.is_monotonic_increasing}")
print(f"Sample of indices:", train_indices[:5])
print(f"Sample of df index:", df.index[:5].tolist())

# After loading data and before splitting
print("\nDetailed Index Analysis:")
print("1. DataFrame properties:")
print(f"- Index type: {type(df.index)}")
print(f"- Index dtype: {df.index.dtype}")
print(f"- Number of unique indices: {len(df.index.unique())}")
print(f"- Any duplicates in index? {df.index.duplicated().any()}")

print("\n2. Loaded indices properties:")
print(f"- Train indices type: {type(train_indices)}")
print(f"- Train indices dtype: {train_indices.dtype}")
print(f"- Number of unique train indices: {len(np.unique(train_indices))}")
print(f"- Any duplicates in train indices? {len(train_indices) != len(np.unique(train_indices))}")

print("\n3. Index overlap analysis:")
print(f"- Train indices in DataFrame index: {np.all(np.isin(train_indices, df.index))}")
print(f"- Val indices in DataFrame index: {np.all(np.isin(val_indices, df.index))}")
print(f"- Test indices in DataFrame index: {np.all(np.isin(test_indices, df.index))}")

# Print a few example rows where indices don't match
missing_indices = train_indices[~np.isin(train_indices, df.index)]
if len(missing_indices) > 0:
    print("\nFirst few missing indices:", missing_indices[:5])

# Split the data using the saved indices with loc instead of iloc
X_train = X.loc[train_indices]
y_train = y.loc[train_indices]
X_val = X.loc[val_indices]
y_val = y.loc[val_indices]
X_test = X.loc[test_indices]
y_test = y.loc[test_indices]

print("\nData split sizes:")
print(f"Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Val: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create XGBoost pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        enable_categorical=True,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        n_jobs=-1,
        max_bin=256,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    ))
])

# Train ensemble of models
n_classifiers = 11
max_depth_values = [2, 3]
n_trees_values = [256, 512]

models = []
for i in range(n_classifiers):
    print(f"\nTraining model {i+1}/{n_classifiers}")
    max_depth = np.random.choice(max_depth_values)
    n_trees = np.random.choice(n_trees_values)
    model = xgb_pipeline.set_params(
        classifier__n_estimators=n_trees,
        classifier__max_depth=max_depth,
        classifier__random_state=42
    )
    model.fit(X_train, y_train)
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
        
        return fpr, tpr, auc_score
    except Exception as e:
        print(f"Error in evaluate_models: {str(e)}")
        raise

# Save models
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
date = datetime.now().strftime('%Y_%m_%d')  # Add date format
model_filename = f'xgboost_ensemble_{date}_run_{timestamp}.joblib'
os.makedirs('saved_models', exist_ok=True)
joblib.dump(models, f'saved_models/{model_filename}')
print(f"Saved trained models as: saved_models/{model_filename}")

# Evaluate and plot results
val_fpr, val_tpr, val_auc = evaluate_models(models, X_val, y_val, "validation")
test_fpr, test_tpr, test_auc = evaluate_models(models, X_test, y_test, "test")

plt.figure(figsize=(10, 8))
plt.plot(val_fpr, val_tpr, color='blue', lw=2, label=f'Validation (AUC = {val_auc:.4f})')
plt.plot(test_fpr, test_tpr, color='red', lw=2, label=f'Test (AUC = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curves - Top 25 XGBoost Features')
plt.legend(loc="lower right")
plt.grid(True)

# Modify the plot saving section
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plot_filename = f'xgboost_roc_curve_{timestamp}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved ROC curve plot as: {plot_filename}")

# FEATURE IMPORTANCE
# After training models...
print("\nCalculating feature importance from trained models...")
# Get the correct shape from the first model
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

# Map feature importance back to original features
preprocessor = models[0].named_steps['preprocessor']
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
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

# Print aggregated feature importance
print("\nFeature Importance (aggregated for categorical features):")
for feat, importance in sorted(aggregated_importance.items(), 
                             key=lambda x: x[1], reverse=True):
    print(f"{feat}: {importance:.4f}")

# LLM-BASED PREDICTION EXPLANATION
'''
!pip install groq
import base64
import groq
import asyncio
import os

def visualize_decision_path(models, sample_idx=0):
    """Create a readable decision tree visualization by setting feature names directly"""
    model = models[0].named_steps['classifier']

    # Get feature names
    preprocessor = models[0].named_steps['preprocessor']
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)

    # Set feature names directly on the booster
    model.get_booster().feature_names = feature_names

    plt.figure(figsize=(25, 15))
    plot_tree(model, num_trees=0)
    plt.tight_layout()
    return plt

# Run visualization
plt = visualize_decision_path(models)
plt.show()

def create_feature_contribution_map(models, X_sample):
    """Create a more readable feature contribution heatmap"""
    # Get feature contributions for each model
    contributions = []

    for model in models:
        X_transformed = model.named_steps['preprocessor'].transform(X_sample)
        contribution = model.named_steps['classifier'].feature_importances_
        contributions.append(contribution)

    # Average contributions across models
    avg_contribution = np.mean(contributions, axis=0)

    # Get feature names
    preprocessor = models[0].named_steps['preprocessor']
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)

    # Sort features by contribution magnitude
    sorted_indices = np.argsort(np.abs(avg_contribution))
    top_n = 20  # Show only top 20 most important features

    selected_indices = sorted_indices[-top_n:]
    selected_contributions = avg_contribution[selected_indices]
    selected_features = [feature_names[i] for i in selected_indices]

    # Create contribution map
    plt.figure(figsize=(15, 8))
    plt.barh(range(len(selected_contributions)), selected_contributions)

    # Customize appearance
    plt.yticks(range(len(selected_features)), selected_features, fontsize=10)
    plt.xlabel('Feature Contribution', fontsize=12)
    plt.title('Top Feature Contributions', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Color positive and negative contributions differently
    colors = ['red' if x < 0 else 'blue' for x in selected_contributions]
    plt.barh(range(len(selected_contributions)), selected_contributions, color=colors)

    # Add value labels on the bars
    for i, v in enumerate(selected_contributions):
        plt.text(v, i, f'{v:.3f}',
                va='center',
                fontsize=8,
                fontweight='bold')

    plt.tight_layout()
    return plt

# Generate and visualize feature contribution map
plt = create_feature_contribution_map(models, X_test.iloc[[0]])
plt.show()

import google.generativeai as genai
from google.colab import userdata
genai.configure(api_key=userdata.get('GEMINI_API_KEY'))

def visualize_decision_path(models, sample_idx):
    """Create a readable decision tree visualization by setting feature names directly"""
    model = models[0].named_steps['classifier']

    # Get feature names
    preprocessor = models[0].named_steps['preprocessor']
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)

    # Set feature names directly on the booster
    model.get_booster().feature_names = feature_names

    plt.figure(figsize=(25, 15))
    plot_tree(model, num_trees=0)
    plt.tight_layout()
    return plt

def save_visualizations(models, X_sample, sample_idx, output_dir='explanation_plots'):
    """Save decision tree and feature importance visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save decision tree plot
    plt = visualize_decision_path(models, sample_idx)
    tree_path = os.path.join(output_dir, 'decision_tree.png')
    plt.savefig(tree_path)
    plt.close()

    # Generate and save feature contribution map
    plt = create_feature_contribution_map(models, X_sample)
    contrib_path = os.path.join(output_dir, 'feature_contribution.png')
    plt.savefig(contrib_path)
    plt.close()

    # Get feature names
    preprocessor = models[0].named_steps['preprocessor']
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)

    return {
        'decision_tree': tree_path,
        'contribution_map': contrib_path
    }, feature_names

def generate_model_explanation(models, X_sample, sample_idx):
    """Generate explanation using Gemini Flash"""
    from google.generativeai import GenerativeModel
    import PIL.Image

    # Generate and save visualizations
    viz_paths, feature_names = save_visualizations(models, X_sample, sample_idx)

    # Get prediction
    y_pred = np.mean([model.predict_proba(X_sample)[:, 1] for model in models])

    # Get feature contributions
    contributions = []
    for model in models:
        X_transformed = model.named_steps['preprocessor'].transform(X_sample)
        contribution = model.named_steps['classifier'].feature_importances_
        contributions.append(contribution)
    avg_contribution = np.mean(contributions, axis=0)

    # Get top features
    sorted_features = sorted(
        zip(feature_names, avg_contribution),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    # Create prompt
    prompt = f"""You are an expert ML model interpreter. You're tasked with explaining an XGBoost model's decision
    path and feature importance plots for malware detection, where the model predicts if an app is malicious (1) or benign (0).

    The model predicted a score of {y_pred:.3f} (higher values indicate higher likelihood of malware).

    The top 5 most influential features were:
    {sorted_features}

    In your response:
    - Explain what features the model is focusing on most heavily based on the feature importance plot
    - Explain what decision path the model took to reach its prediction, based on the decision tree visualization
    - Explain possible reasons why the model made the prediction it did
    - Keep your explanation to 4 sentences max"""

    # Load images
    tree_img = PIL.Image.open(viz_paths['decision_tree'])
    feat_img = PIL.Image.open(viz_paths['contribution_map'])

    # Get explanation from Gemini
    model = GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content([prompt, tree_img, feat_img])

    return response.text, y_pred

def explain_prediction(models, X_sample, sample_idx):
    """Wrapper to get and print model explanation"""
    try:
        explanation, y_pred = generate_model_explanation(models, X_sample, sample_idx)
        print(f"Prediction: {y_pred:.3f}")
        print("\nModel Explanation:")
        print("-" * 80)
        # print(explanation)
        return explanation, y_pred
    except Exception as e:
        print(f"Error getting explanation: {e}")
        return None, None

# Get explanation for a sample
sample_idx = 1
# print(X_train.iloc[sample_idx])
X_sample = X_test.iloc[[sample_idx]]
explain_prediction(models, X_sample, sample_idx)
'''

# Add at the very end of the script
sys.stdout = sys.stdout.terminal  # Restore normal stdout
print(f"Training log saved to: {log_filename}")