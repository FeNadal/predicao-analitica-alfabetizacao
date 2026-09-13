# -*- coding: utf-8 -*-
"""
Módulo de Treinamento, Otimização e Validação de Modelos Supervisionados
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)

from src.preprocessing.data_loader import load_data
from src.preprocessing.features import engineer_features, get_feature_names
from src.preprocessing.pipeline import build_ml_pipeline
from src.evaluation.metrics import evaluate_classification, evaluate_regression

def train_and_evaluate_all() -> Dict[str, Any]:
    print("1. Carregando dados da camada Gold...")
    df_raw = load_data()
    
    print("2. Aplicando engenharia de atributos...")
    df_feat = engineer_features(df_raw)
    
    num_cols, cat_cols = get_feature_names()
    feature_cols = num_cols + cat_cols
    
    # Separação temporal estrita: Treino em 2023, Teste em 2024
    df_2023 = df_feat[df_feat["ano"] == 2023].dropna(subset=["taxa_alfabetizacao"]).copy()
    df_2024 = df_feat[df_feat["ano"] == 2024].dropna(subset=["taxa_alfabetizacao"]).copy()
    print(f"   -> Treino (2023): {len(df_2023):,} | Teste (2024): {len(df_2024):,}")
    
    X_train = df_2023[feature_cols]
    y_clf_train = df_2023["target_risco_educacional"]
    y_reg_train = df_2023["target_taxa_alfabetizacao"]
    
    X_test = df_2024[feature_cols]
    y_clf_test = df_2024["target_risco_educacional"]
    y_reg_test = df_2024["target_taxa_alfabetizacao"]
    
    # ----------------------------------------------------
    # CLASSIFICAÇÃO DE RISCO EDUCACIONAL
    # ----------------------------------------------------
    print("\n3. Treinando e avaliando Modelos de Classificação...")
    clf_models = {
        "Regressão Logística (Baseline)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting Classifier": HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, min_samples_leaf=15, random_state=42
        )
    }
    
    cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_results = {}
    best_clf_score = -1
    best_clf_name = None
    best_clf_pipe = None
    
    for name, model in clf_models.items():
        pipe = build_ml_pipeline(model, num_cols, cat_cols)
        cv_scores = cross_validate(
            pipe, X_train, y_clf_train, cv=cv_clf,
            scoring=["accuracy", "balanced_accuracy", "f1", "roc_auc"]
        )
        pipe.fit(X_train, y_clf_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        test_eval = evaluate_classification(y_clf_test, y_pred, y_prob)
        
        clf_results[name] = {
            "cv_mean_roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_mean_f1": float(np.mean(cv_scores["test_f1"])),
            "cv_mean_bal_acc": float(np.mean(cv_scores["test_balanced_accuracy"])),
            "test_accuracy": test_eval["accuracy"],
            "test_balanced_accuracy": test_eval["balanced_accuracy"],
            "test_precision": test_eval["precision"],
            "test_recall": test_eval["recall"],
            "test_f1_score": test_eval["f1_score"],
            "test_roc_auc": test_eval["roc_auc"],
            "confusion_matrix": test_eval["confusion_matrix"]
        }
        print(f"   [{name}] CV ROC-AUC: {clf_results[name]['cv_mean_roc_auc']:.4f} | Teste ROC-AUC: {test_eval['roc_auc']:.4f} | Teste F1: {test_eval['f1_score']:.4f}")
        if test_eval["roc_auc"] > best_clf_score:
            best_clf_score = test_eval["roc_auc"]
            best_clf_name = name
            best_clf_pipe = pipe
            
    # ----------------------------------------------------
    # REGRESSÃO DA TAXA DE ALFABETIZAÇÃO
    # ----------------------------------------------------
    print("\n4. Treinando e avaliando Modelos de Regressão...")
    reg_models = {
        "Ridge Regression (Baseline)": Ridge(alpha=10.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(
            max_iter=150, max_depth=6, min_samples_leaf=15, random_state=42
        )
    }
    
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_results = {}
    best_reg_score = -999
    best_reg_name = None
    best_reg_pipe = None
    
    for name, model in reg_models.items():
        pipe = build_ml_pipeline(model, num_cols, cat_cols)
        cv_scores = cross_validate(
            pipe, X_train, y_reg_train, cv=cv_reg,
            scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        )
        pipe.fit(X_train, y_reg_train)
        y_pred = pipe.predict(X_test)
        test_eval = evaluate_regression(y_reg_test, y_pred)
        
        reg_results[name] = {
            "cv_mean_r2": float(np.mean(cv_scores["test_r2"])),
            "cv_mean_mae": float(-np.mean(cv_scores["test_neg_mean_absolute_error"])),
            "cv_mean_rmse": float(-np.mean(cv_scores["test_neg_root_mean_squared_error"])),
            "test_r2": test_eval["r2"],
            "test_mae": test_eval["mae"],
            "test_rmse": test_eval["rmse"],
            "test_mape": test_eval["mape"]
        }
        print(f"   [{name}] CV R2: {reg_results[name]['cv_mean_r2']:.4f} | Teste R2: {test_eval['r2']:.4f} | Teste MAE: {test_eval['mae']:.2f}% | Teste RMSE: {test_eval['rmse']:.2f}%")
        if test_eval["r2"] > best_reg_score:
            best_reg_score = test_eval["r2"]
            best_reg_name = name
            best_reg_pipe = pipe
            
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(best_clf_pipe, "data/processed/best_classification_model.joblib")
    joblib.dump(best_reg_pipe, "data/processed/best_regression_model.joblib")
    
    df_preds = df_2024[["id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "taxa_alfabetizacao", "meta_alfabetizacao_2030"]].copy()
    df_preds["taxa_real_2024"] = y_reg_test
    df_preds["taxa_prevista_2024"] = best_reg_pipe.predict(X_test)
    df_preds["erro_predicao"] = df_preds["taxa_real_2024"] - df_preds["taxa_prevista_2024"]
    df_preds["risco_real_2024"] = y_clf_test
    df_preds["risco_predito_2024"] = best_clf_pipe.predict(X_test)
    df_preds["probabilidade_risco"] = best_clf_pipe.predict_proba(X_test)[:, 1]
    df_preds.to_csv("data/processed/test_predictions_2024.csv", index=False, encoding="utf-8")
    
    summary = {
        "best_classification_model": best_clf_name,
        "best_regression_model": best_reg_name,
        "classification_results": clf_results,
        "regression_results": reg_results
    }
    with open("data/processed/evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    print("\n5. Modelagem concluída com sucesso!")
    print(f"   Melhor Classificador: {best_clf_name} (ROC-AUC: {best_clf_score:.4f})")
    print(f"   Melhor Regressor:     {best_reg_name} (R2: {best_reg_score:.4f})")
    return summary

if __name__ == "__main__":
    train_and_evaluate_all()
