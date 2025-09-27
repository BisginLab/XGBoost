# XGBoost Compute Profile Averaging (MI-25 & FI-25)

## Overview

This feature allows you to profile XGBoost models with both MI-25 (Mutual Information) and FI-25 (Feature Importance) feature sets, then average the results into a single compute table per model/sample_size/device.

## Usage

### 1. Profile MI-25 Models

Run profiling for each size on CPU (fair comparison):

```bash
# MI-25 (standardized features)
python profile_xgboost.py \
  --models '{"10000":"/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_10000_run_20250825_160615.joblib","100000":"/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_100000_run_20250825_164742.joblib","full":"/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_full_run_20250825_165926.joblib"}' \
  --df_path ./content/sample_data/corrected_permacts.csv \
  --indices_dir ./standardized_data \
  --device cpu \
  --single_model \
  --feature_set MI-25
```

### 2. Profile FI-25 Models

```bash
# FI-25 (feature-importance features)
python profile_xgboost.py \
  --models '{"10000":"saved_models/xgboost_ensemble_fi_features_10000_run_20250828_061856.joblib","100000":"saved_models/xgboost_ensemble_fi_features_100000_run_20250828_065720.joblib","full":"saved_models/xgboost_ensemble_fi_features_full_run_20250828_071750.joblib"}' \
  --df_path ./content/sample_data/corrected_permacts.csv \
  --indices_dir ./standardized_data \
  --device cpu \
  --single_model \
  --feature_set FI-25
```

### 3. Build Averaged Table

```bash
# Build averaged table (collapses MI & FI)
python scripts/make_compute_table_avg.py
```

## Outputs

- **Individual profiles**: `compute_profiles/XGB-infer-{size}_{timestamp}.json` (tagged with feature_set)
- **Combined CSV**: `compute_profiles/compute_profiles_summary.csv` (includes feature_set column)
- **Averaged table**: `compute_profiles/compute_profiles_avg.csv` (one row per {model, sample_size, device})
- **LaTeX table**: `compute_profiles/compute_profiles_avg.tex` (for papers)

## Averaging Policy

- **Time/throughput**: Mean over MI & FI
- **Memory peaks**: Max over MI & FI (conservative)
- **Model size**: Mean over MI & FI
- **Standard deviation**: Included when both regimes exist

## Example Output

The averaged table will have one row per combination:

| model | sample_size | device | test_time_s | test_time_s_std | throughput_apps_per_s | throughput_std | cpu_max_rss_mb | peak_vram_mb | model_bytes |
|-------|-------------|---------|-------------|------------------|---------------------|----------------|----------------|--------------|-------------|
| XGBoost | 10000 | cpu | 0.245 | 0.012 | 4081.633 | 203.456 | 781.2 | 0 | 52428800 |
| XGBoost | 100000 | cpu | 2.156 | 0.089 | 464.286 | 19.123 | 1245.8 | 0 | 52428800 |
| XGBoost | full | cpu | 18.432 | 0.456 | 54.286 | 1.345 | 2156.4 | 0 | 52428800 |
