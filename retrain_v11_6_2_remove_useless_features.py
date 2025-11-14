#!/usr/bin/env python3
"""
Retrain V11.6.2: Remove useless features.

Remove:
- confidence_score_market (0.0% importance, no variance)

Keep all other features including timing_score and competitor_count
despite low importance, as they were part of the normalization fix.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from datetime import datetime

# V11.6.2 features (17 features - removed confidence_score_market)
V11_6_2_FEATURES = [
    'l_level', 'estimated_age', 'founder_experience_score', 'timing_score',
    'market_size_billion', 'cagr_percent', 'competitor_count', 'market_maturity_stage',
    'confidence_score', 'geographic_advantage',  # Removed: confidence_score_market
    'description_sentiment', 'description_complexity', 'about_quality',
    'sector_keyword_score', 'founder_market_fit', 'market_saturation_score',
    'differentiation_score'
]

def normalize_features_full(df):
    """Normalize features to match current extraction pipeline."""
    df_fixed = df.copy()
    
    # Fix 1: timing_score (0-5 → 0-1)
    if 'timing_score' in df_fixed.columns:
        df_fixed['timing_score'] = df_fixed['timing_score'] / 5.0
    
    # Fix 2: market_size_billion (TAM → SAM)
    if 'market_size_billion' in df_fixed.columns:
        df_fixed['market_size_billion'] = df_fixed['market_size_billion'] / 19.0
    
    # Fix 3: competitor_count (broad → direct)
    if 'competitor_count' in df_fixed.columns:
        df_fixed['competitor_count'] = df_fixed['competitor_count'] / 4.6
    
    return df_fixed

def main():
    print("="*80)
    print("V11.6.2 RETRAINING - REMOVE USELESS FEATURES")
    print("="*80)
    print("Removed: confidence_score_market (0.0% importance)")
    print("Features: 17 (was 18)")
    print()
    
    # Load training data
    print("Loading training data...")
    df = pd.read_csv('final_years_with_phase3_features_20251103_195924.csv')
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Normalize features
    df = normalize_features_full(df)
    
    # Ensure 'founded_year' is numeric
    if 'founded year' in df.columns:
        df['founded_year'] = pd.to_numeric(df['founded year'], errors='coerce')
    elif 'founded_year' in df.columns:
        df['founded_year'] = pd.to_numeric(df['founded_year'], errors='coerce')
    else:
        print("❌ Error: 'founded year' column not found.")
        return

    df = df.dropna(subset=['founded_year'])

    # Convert l_level
    if 'l_level' in df.columns:
        df['l_level'] = df['l_level'].astype(str).str.replace('L', '').str.replace('INSUFFICIENT_DATA', '2')
        df['l_level'] = pd.to_numeric(df['l_level'], errors='coerce')
        df['l_level'] = df['l_level'].fillna(2.0)
    
    # Convert Success to binary
    df['Success_binary'] = (df['Success'] == 'yes').astype(int)
    
    # Check features
    print("\nChecking features...")
    available = []
    missing = []
    for feat in V11_6_2_FEATURES:
        if feat in df.columns:
            available.append(feat)
        else:
            missing.append(feat)
    
    print(f"✅ Available: {len(available)}/{len(V11_6_2_FEATURES)}")
    if missing:
        print(f"⚠️ Missing: {missing}")
        for feat in missing:
            df[feat] = 0.7 if feat == 'confidence_score' else 0.0
    
    # Time-based split (2025 optimized)
    print("\n" + "="*80)
    print("TIME-BASED DATA SPLIT (2025 OPTIMIZED)")
    print("="*80)

    train_years = range(2021, 2025)
    val_years = range(2024, 2025)
    test_years = range(2023, 2025)

    train_df = df[df['founded_year'].isin(train_years)].copy()
    val_df = df[df['founded_year'].isin(val_years)].copy()
    test_df = df[df['founded_year'].isin(test_years)].copy()

    print(f"\nTraining set ({min(train_years)}-{max(train_years)}):")
    print(f"  Companies: {len(train_df):,}")
    print(f"  Success rate: {train_df['Success_binary'].mean()*100:.1f}%")

    print(f"\nValidation set ({min(val_years)}-{max(val_years)}):")
    print(f"  Companies: {len(val_df):,}")
    print(f"  Success rate: {val_df['Success_binary'].mean()*100:.1f}%")

    print(f"\nTest set ({min(test_years)}-{max(test_years)}):")
    print(f"  Companies: {len(test_df):,}")
    print(f"  Success rate: {test_df['Success_binary'].mean()*100:.1f}%")

    # Prepare data
    print("\nPreparing data...")
    train_df_clean = train_df.dropna(subset=V11_6_2_FEATURES + ['Success_binary'])
    val_df_clean = val_df.dropna(subset=V11_6_2_FEATURES + ['Success_binary'])
    test_df_clean = test_df.dropna(subset=V11_6_2_FEATURES + ['Success_binary'])

    X_train = train_df_clean[V11_6_2_FEATURES].values
    y_train = train_df_clean['Success_binary'].values
    X_val = val_df_clean[V11_6_2_FEATURES].values
    y_val = val_df_clean['Success_binary'].values
    X_test = test_df_clean[V11_6_2_FEATURES].values
    y_test = test_df_clean['Success_binary'].values

    print(f"✅ Clean datasets:")
    print(f"   Train: {len(X_train)} samples ({sum(y_train)/len(y_train)*100:.1f}% success)")
    print(f"   Val:   {len(X_val)} samples ({sum(y_val)/len(y_val)*100:.1f}% success)")
    print(f"   Test:  {len(X_test)} samples ({sum(y_test)/len(y_test)*100:.1f}% success)")

    # Cap outliers
    print("\nCapping outliers at 99th percentile...")
    for i, feat in enumerate(V11_6_2_FEATURES):
        if feat in train_df_clean.columns:
            p99 = train_df_clean[feat].quantile(0.99)
            if pd.notna(p99):
                X_train[:, i] = np.clip(X_train[:, i], None, p99)
                X_val[:, i] = np.clip(X_val[:, i], None, p99)
                X_test[:, i] = np.clip(X_test[:, i], None, p99)

    # Normalize
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("\n" + "="*80)
    print("TRAINING XGBOOST...")
    print("="*80)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("PERFORMANCE")
    print("="*80)
    
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])
    val_roc_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    print(f"\nAccuracy:")
    print(f"  Train: {train_acc*100:.1f}%")
    print(f"  Val:   {val_acc*100:.1f}%")
    print(f"  Test:  {test_acc*100:.1f}%")

    print(f"\nROC AUC:")
    print(f"  Train: {train_roc_auc:.3f}")
    print(f"  Val:   {val_roc_auc:.3f}")
    print(f"  Test:  {test_roc_auc:.3f}")
    
    print("\nTest Set Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Failure', 'Success']))
    
    # Feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (17 features)")
    print("="*80)
    feature_importance = model.feature_importances_
    sorted_features = sorted(
        zip(V11_6_2_FEATURES, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feat, imp) in enumerate(sorted_features, 1):
        pct = imp / sum(feature_importance) * 100
        bar = '█' * int(pct / 2)
        print(f"{i:2d}. {feat:30s}: {imp:.4f} ({pct:5.1f}%) {bar}")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'v11_6_2_model_{timestamp}.pkl'
    scaler_path = f'v11_6_2_scaler_{timestamp}.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n" + "="*80)
    print("SAVED - V11.6.2 MODEL (CLEANER)")
    print("="*80)
    print(f"✅ Model:  {model_path}")
    print(f"✅ Scaler: {scaler_path}")
    print(f"✅ Test Accuracy: {test_acc*100:.1f}%")
    print(f"✅ Test ROC AUC: {test_roc_auc:.3f}")
    print(f"✅ Features: {len(V11_6_2_FEATURES)} (removed confidence_score_market)")
    print(f"✅ Training Period: 2021-2024")
    
    return model_path, scaler_path

if __name__ == "__main__":
    main()

