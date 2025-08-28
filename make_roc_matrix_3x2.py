# make_roc_matrix_3x2.py
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from category_encoders import CatBoostEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============
# CONFIG — EDIT THESE PATHS FOR YOUR MACHINE
# ============
CSV_PATH = "/home/umflint.edu/koernerg/xgboost/content/sample_data/corrected_permacts.csv"
INDICES_DIR = "/home/umflint.edu/koernerg/excelformer/ExcelFormer/standardized_data"
MI_JSON_PATH = "/home/umflint.edu/koernerg/excelformer/ExcelFormer/mi_top25_android_security.json"  # <- put your saved MI JSON here if you have it

# ExcelFormer repo so we can import the model class
EXCELFORMER_REPO = "/home/umflint.edu/koernerg/excelformer/ExcelFormer"

# ExcelFormer checkpoints (trained with the top-25)
EF_CHECKPOINTS = {
    "10000": "/home/umflint.edu/koernerg/excelformer/ExcelFormer/result/ExcelFormer/default/mixup(none)/android_security/42/10000/pytorch_model.pt",
    "100000": "/home/umflint.edu/koernerg/excelformer/ExcelFormer/result/ExcelFormer/default/mixup(none)/android_security/42/100000/pytorch_model.pt",
    "full": "/home/umflint.edu/koernerg/excelformer/ExcelFormer/result/ExcelFormer/default/mixup(none)/android_security/42/full/pytorch_model.pt",
}

# XGBoost saved pipelines / ensembles
XGB_MODELS = {
    "10000": "/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_10000_run_20250824_170708.joblib",
    "100000": "/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_100000_run_20250824_173133.joblib",
    "full": "/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_full_run_20250824_175712.joblib",
}

OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============
# FALLBACK: your 25 MI features (exact order) if JSON not found
# ============
MI_TOP25_FALLBACK = [
    "ContentRating", "LastUpdated", "days_since_last_update",
    "highest_android_version", "privacy_policy_link", "TwoStarRatings",
    "CurrentVersion", "isSpamming", "max_downloads_log", "OneStarRatings",
    "FourStarRatings", "ThreeStarRatings", "lowest_android_version", "STORAGE",
    "FiveStarRatings", "LenWhatsNew", "AndroidVersion", "developer_address",
    "developer_website", "intent", "PHONE", "LOCATION", "DeveloperCategory",
    "ReviewsAverage", "Genre",
]

# ============
# Wire up ExcelFormer model import
# ============
sys.path.append(EXCELFORMER_REPO)
try:
    from bin import ExcelFormer
except Exception as e:
    raise RuntimeError(
        f"Could not import ExcelFormer from {EXCELFORMER_REPO}. "
        f"Make sure the path is correct and contains the 'bin' package."
    ) from e


def load_selected_features(mi_json_path: str) -> List[str]:
    if mi_json_path and os.path.isfile(mi_json_path):
        with open(mi_json_path, "r") as f:
            payload = json.load(f)
        # accept either a dict like {"selected_features":[...]} or a plain list
        if isinstance(payload, dict) and "selected_features" in payload:
            feats = payload["selected_features"]
        elif isinstance(payload, list):
            feats = payload
        else:
            feats = MI_TOP25_FALLBACK
        print("[INFO] Loaded selected features from JSON:", feats)
        return feats
    print("[WARN] MI JSON not found. Falling back to hard-coded top-25.")
    return MI_TOP25_FALLBACK


def load_indices(size: str, base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = size if size in ("10000", "100000") else "full"
    tr = np.load(os.path.join(base_dir, f"train_indices_{tag}.npy"))
    va = np.load(os.path.join(base_dir, f"val_indices_{tag}.npy"))
    te = np.load(os.path.join(base_dir, f"test_indices_{tag}.npy"))
    return tr, va, te


def split_df(df: pd.DataFrame, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> Dict[str, pd.DataFrame]:
    # Indices are row indices from the original dataset
    return {
        "train": df.iloc[tr],
        "val": df.iloc[va],
        "test": df.iloc[te],
    }


def prepare_ef_matrices(
    df_splits: Dict[str, pd.DataFrame],
    selected_features: List[str],
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For ExcelFormer: CatBoost-encode categorical columns (fit on train only),
    concatenate with numeric, then (optionally) apply QuantileTransformer fit on train.
    Returns: X_val, y_val, X_test, y_test
    """
    for part in ("train", "val", "test"):
        assert all(f in df_splits[part].columns for f in selected_features), \
            f"Missing features in {part} split."

    # target
    y_train = df_splits["train"]["status"].values.astype(np.float32)
    y_val = df_splits["val"]["status"].values.astype(np.float32)
    y_test = df_splits["test"]["status"].values.astype(np.float32)

    # subset features in order
    Xtr = df_splits["train"][selected_features].copy()
    Xva = df_splits["val"][selected_features].copy()
    Xte = df_splits["test"][selected_features].copy()

    # infer cats by dtype object
    cat_cols = [c for c in selected_features if Xtr[c].dtype == "object"]
    num_cols = [c for c in selected_features if c not in cat_cols]

    # CatBoostEncoder on train cats
    if len(cat_cols) > 0:
        cbe = CatBoostEncoder(cols=cat_cols, return_df=False)
        cbe.fit(Xtr[cat_cols], y_train)
        tr_cat = cbe.transform(Xtr[cat_cols]).astype(np.float32)
        va_cat = cbe.transform(Xva[cat_cols]).astype(np.float32)
        te_cat = cbe.transform(Xte[cat_cols]).astype(np.float32)

        tr_num = Xtr[num_cols].to_numpy(dtype=np.float32) if num_cols else np.zeros((len(Xtr), 0), np.float32)
        va_num = Xva[num_cols].to_numpy(dtype=np.float32) if num_cols else np.zeros((len(Xva), 0), np.float32)
        te_num = Xte[num_cols].to_numpy(dtype=np.float32) if num_cols else np.zeros((len(Xte), 0), np.float32)

        X_train = np.concatenate([tr_cat, tr_num], axis=1)
        X_val = np.concatenate([va_cat, va_num], axis=1)
        X_test = np.concatenate([te_cat, te_num], axis=1)
    else:
        # no cats — pure numeric
        X_train = Xtr.to_numpy(dtype=np.float32)
        X_val = Xva.to_numpy(dtype=np.float32)
        X_test = Xte.to_numpy(dtype=np.float32)

    # Optional: same normalization used in training (quantile→normal)
    if normalize:
        n_quantiles = max(min(X_train.shape[0] // 30, 1000), 10)
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=1e9,
            random_state=0,
        )
        qt.fit(X_train)
        X_train = qt.transform(X_train).astype(np.float32)
        X_val = qt.transform(X_val).astype(np.float32)
        X_test = qt.transform(X_test).astype(np.float32)

    return X_val, y_val, X_test, y_test


@torch.inference_mode()
def ef_predict_proba(checkpoint_path: str, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Validate feature count if saved
    expected = ckpt.get("n_features", None)
    if expected is not None and expected != X_test.shape[1]:
        raise RuntimeError(
            f"ExcelFormer checkpoint expects {expected} features; got {X_test.shape[1]}"
        )

    # Construct model to match training config
    model = ExcelFormer(
        d_numerical=X_test.shape[1],
        d_out=2,
        categories=None,
        token_bias=True,
        n_layers=3,
        n_heads=32,
        d_token=256,
        attention_dropout=0.3,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        prenormalization=True,
        kv_compression=None,
        kv_compression_sharing=None,
        init_scale=0.01,
    ).to(device).float()

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def _batched_pred(X: np.ndarray, bs: int = 2048) -> np.ndarray:
        ds = TensorDataset(torch.from_numpy(X).to(device))
        dl = DataLoader(ds, batch_size=bs, shuffle=False)
        out = []
        for (x,) in dl:
            logits = model(x, None)
            # logits shape: [B, 2]
            probs = torch.softmax(logits, dim=1)[:, 1]
            out.append(probs.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    p_val = _batched_pred(X_val)
    p_test = _batched_pred(X_test)
    return p_val, p_test


def xgb_predict_proba(model_artifact, X_val_df: pd.DataFrame, X_test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts either a single Pipeline or a list of Pipelines (ensemble).
    Each pipeline should be (preprocessor -> classifier).
    """
    if isinstance(model_artifact, list):
        models = model_artifact
    else:
        models = [model_artifact]

    def _avg_proba(models, X_df: pd.DataFrame) -> np.ndarray:
        preds = []
        for m in models:
            try:
                # If it's a pipeline, just predict_proba on the DataFrame
                y = m.predict_proba(X_df)[:, 1]
            except Exception:
                # Fallback: try manual transform if attributes present
                pre = m.named_steps.get("preprocessor", None)
                clf = m.named_steps.get("classifier", None)
                if pre is None or clf is None:
                    raise
                Xt = pre.transform(X_df)
                # Force CPU prediction if param is available
                try:
                    clf.set_params(device="cpu")
                except Exception:
                    pass
                y = clf.predict_proba(Xt)[:, 1]
            preds.append(y)
        return np.mean(np.stack(preds, axis=0), axis=0)

    p_val = _avg_proba(models, X_val_df)
    p_test = _avg_proba(models, X_test_df)
    return p_val, p_test


def safe_load_joblib(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = joblib.load(path)
    return obj


def compute_curves(y_true_val, p_val, y_true_test, p_test):
    fpr_v, tpr_v, _ = roc_curve(y_true_val, p_val)
    fpr_t, tpr_t, _ = roc_curve(y_true_test, p_test)
    auc_v = roc_auc_score(y_true_val, p_val)
    auc_t = roc_auc_score(y_true_test, p_test)
    return (fpr_v, tpr_v, auc_v), (fpr_t, tpr_t, auc_t)


def main():
    sizes = ["10000", "100000", "full"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    # dropna to mirror earlier preprocessing
    df = df.dropna()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    assert "status" in df.columns, "Target column 'status' not found."

    # Selected features (names + order)
    selected_features = load_selected_features(MI_JSON_PATH)
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise RuntimeError(f"Selected features missing in CSV: {missing}")

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    for col, size in enumerate(sizes):
        # indices
        tr_idx, va_idx, te_idx = load_indices(size, INDICES_DIR)
        splits = split_df(df, tr_idx, va_idx, te_idx)

        # --------- ExcelFormer (row 0) ----------
        ax = axes[0, col]
        title = f"ExcelFormer — {size}"
        try:
            X_val, y_val, X_test, y_test = prepare_ef_matrices(
                {
                    "train": splits["train"],
                    "val": splits["val"],
                    "test": splits["test"],
                },
                selected_features,
                normalize=True,
            )
            ef_ckpt = EF_CHECKPOINTS[size]
            p_val, p_test = ef_predict_proba(ef_ckpt, X_val, X_test)
            (fpr_v, tpr_v, auc_v), (fpr_t, tpr_t, auc_t) = compute_curves(y_val, p_val, y_test, p_test)

            ax.plot(fpr_v, tpr_v, lw=2, label=f"Val AUC = {auc_v:.3f}")
            ax.plot(fpr_t, tpr_t, lw=2, label=f"Test AUC = {auc_t:.3f}")
            ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
            ax.set_title(title)
            ax.grid(True, alpha=0.4)
            ax.legend(loc="lower right", frameon=True)
        except Exception as e:
            ax.text(0.5, 0.5, f"Missing/failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
            ax.grid(True, alpha=0.4)

        # --------- XGBoost (row 1) ----------
        ax = axes[1, col]
        title = f"XGBoost — {size}"
        try:
            xgb_path = XGB_MODELS[size]
            model_art = safe_load_joblib(xgb_path)

            # Feed original feature columns (the pipeline will do its own transforms)
            X_val_df = splits["val"][selected_features]
            X_test_df = splits["test"][selected_features]
            y_val = splits["val"]["status"].values
            y_test = splits["test"]["status"].values

            p_val, p_test = xgb_predict_proba(model_art, X_val_df, X_test_df)
            (fpr_v, tpr_v, auc_v), (fpr_t, tpr_t, auc_t) = compute_curves(y_val, p_val, y_test, p_test)

            ax.plot(fpr_v, tpr_v, lw=2, label=f"Val AUC = {auc_v:.3f}")
            ax.plot(fpr_t, tpr_t, lw=2, label=f"Test AUC = {auc_t:.3f}")
            ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
            ax.set_title(title)
            ax.grid(True, alpha=0.4)
            ax.legend(loc="lower right", frameon=True)
        except Exception as e:
            ax.text(0.5, 0.5, f"Missing/failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
            ax.grid(True, alpha=0.4)

    # Shared labels
    for r in range(2):
        axes[r, 0].set_ylabel("True Positive Rate")
    for c in range(3):
        axes[1, c].set_xlabel("False Positive Rate")

    fig.suptitle("ROC Curves — ExcelFormer vs XGBoost (Val & Test) @ {10k, 100k, Full}", y=0.98)
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])

    out_png = os.path.join(OUTPUT_DIR, f"roc_matrix_3x2_{ts}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"roc_matrix_3x2_{ts}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_pdf}")


if __name__ == "__main__":
    main()
