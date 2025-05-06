import numpy as np
import pandas as pd
import os

def load_consistent_samples(df, sample_size, indices_dir='sample_indices'):
    """
    Load consistent samples for training across different models.
    
    Args:
        df (pd.DataFrame): The full dataset
        sample_size (int): Size of sample to load (5000, 10000, or 100000)
        indices_dir (str): Directory containing saved indices
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test) containing the train/val/test splits
    """
    if sample_size not in [5000, 10000, 100000]:
        raise ValueError("sample_size must be one of: 5000, 10000, 100000")
    
    # Load the saved indices
    train_indices_path = os.path.join(indices_dir, f'train_indices_{sample_size}.npy')
    val_indices_path = os.path.join(indices_dir, f'val_indices_{sample_size}.npy')
    test_indices_path = os.path.join(indices_dir, f'test_indices_{sample_size}.npy')
    
    if not all(os.path.exists(p) for p in [train_indices_path, val_indices_path, test_indices_path]):
        raise ValueError(f"No saved indices found for size {sample_size}. Please run train_xgboost.py first to generate indices.")
    
    train_indices = np.load(train_indices_path)
    val_indices = np.load(val_indices_path)
    test_indices = np.load(test_indices_path)
    
    # Get features and target
    X = df.drop('status', axis=1)
    y = df['status']
    
    # Return sampled data with train/val/test splits
    return (X.loc[train_indices], y.loc[train_indices],
            X.loc[val_indices], y.loc[val_indices],
            X.loc[test_indices], y.loc[test_indices])

def verify_sample_consistency(df1, df2, sample_size):
    """
    Verify that two datasets contain the exact same samples.
    
    Args:
        df1 (pd.DataFrame): First dataset
        df2 (pd.DataFrame): Second dataset
        sample_size (int): Expected sample size
    
    Returns:
        bool: True if datasets are consistent, False otherwise
    """
    if len(df1) != len(df2) or len(df1) != sample_size:
        return False
    
    # Check if indices match
    return df1.index.equals(df2.index)

if __name__ == "__main__":
    # Example usage
    print("This is a utility module for loading consistent samples across models.")
    print("Import and use the load_consistent_samples() function in your model training scripts.") 