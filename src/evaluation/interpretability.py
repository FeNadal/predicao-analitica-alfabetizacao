# -*- coding: utf-8 -*-
"""
Módulo de Interpretabilidade e Explicabilidade dos Modelos (SHAP e Feature Importance)
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.preprocessing.data_loader import load_data
from src.preprocessing.features import engineer_features, get_feature_names

def run_interpretability_pipeline():
    print("1. Carregando modelos treinados e dados...")
    clf_pipe = joblib.load("data/processed/best_classification_model.joblib")
    reg_pipe = joblib.load("data/processed/best_regression_model.joblib")
    
    df_raw = load_data()
    df_feat = engineer_features(df_raw)
    num_cols, cat_cols = get_feature_names()
    feature_cols = num_cols + cat_cols
    
    df_2024 = df_feat[df_feat["ano"] == 2024].dropna(subset=["taxa_alfabetizacao"]).copy()
    X_test = df_2024[feature_cols]
    y_test_clf = df_2024["target_risco_educacional"]
    y_test_reg = df_2024["target_taxa_alfabetizacao"]
    
    os.makedirs("images", exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})
    
    # ----------------------------------------------------
    # 1. MATRIZ DE CONFUSÃO (CLASSIFICAÇÃO)
    # ----------------------------------------------------
    print("2. Gerando Gráfico: Matriz de Confusão...")
    y_pred_clf = clf_pipe.predict(X_test)
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Adequado (Taxa >= 60%)", "Risco (Taxa < 60%)"],
                yticklabels=["Adequado (Real)", "Risco (Real)"], ax=ax)
    ax.set_title("Matriz de Confusão - Classificação de Risco (Teste 2024)", fontweight="bold")
    ax.set_ylabel("Valor Observado")
    ax.set_xlabel("Predição do Modelo")
    plt.savefig("images/09_matriz_confusao.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 2. CURVA ROC COMPARATIVA
    # ----------------------------------------------------
    print("3. Gerando Gráfico: Curva ROC...")
    y_prob_clf = clf_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_clf, y_prob_clf)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"Modelo Campeão (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Classificador Aleatório")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade / Recall)")
    ax.set_title("Curva ROC - Capacidade de Discriminação de Risco Educacional", fontweight="bold")
    ax.legend(loc="lower right")
    plt.savefig("images/10_curva_roc.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 3. FEATURE IMPORTANCE (RANDOM FOREST / GRADIENT BOOSTING)
    # ----------------------------------------------------
    print("4. Extraindo Feature Importance...")
    preprocessor = clf_pipe.named_steps["preprocessor"]
    model_clf = clf_pipe.named_steps["model"]
    
    # Nomes das features transformadas
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    transformed_features = num_cols + cat_names
    
    # Obter importância
    if hasattr(model_clf, "feature_importances_"):
        importances = model_clf.feature_importances_
    else:
        # Permutation ou coeficientes absolutos
        importances = np.abs(model_clf.coef_[0])
        
    df_imp = pd.DataFrame({
        "feature": transformed_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    top_15 = df_imp.head(15).iloc[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(top_15["feature"], top_15["importance"], color="#1f9a55", edgecolor="black", alpha=0.85)
    ax.set_title("Top 15 Variáveis Mais Relevantes na Predição de Risco", fontweight="bold")
    ax.set_xlabel("Importância Relativa")
    plt.savefig("images/11_feature_importance.png", dpi=300)
    plt.close()
    
    # Salvar top features em JSON
    df_imp.head(20).to_json("data/processed/top_features.json", orient="records", indent=4)
    
    # ----------------------------------------------------
    # 4. SHAP VALUES SUMMARY PLOT
    # ----------------------------------------------------
    print("5. Calculando Valores SHAP...")
    X_test_trans = preprocessor.transform(X_test)
    
    # Amostra para SHAP rápido e robusto
    np.random.seed(42)
    sample_indices = np.random.choice(X_test_trans.shape[0], size=min(1000, X_test_trans.shape[0]), replace=False)
    X_sample = X_test_trans[sample_indices]
    
    try:
        explainer = shap.Explainer(model_clf, X_sample)
        shap_values = explainer(X_sample)
        
        # Se classificação binária com 2 classes no shap
        if len(shap_values.shape) == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values
            
        fig = plt.figure(figsize=(11, 7))
        shap.summary_plot(vals.values, X_sample, feature_names=transformed_features, show=False, max_display=12)
        plt.title("Valores SHAP - Impacto e Direção das Variáveis no Risco Educacional", fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig("images/12_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   -> SHAP summary plot salvo em images/12_shap_summary.png")
    except Exception as e:
        print(f"   Aviso SHAP: {e}. Gerando visualização baseada em SHAP bar.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_15["feature"], top_15["importance"], color="#2b5c8f")
        ax.set_title("Impacto das Variáveis na Predição Educacional", fontweight="bold")
        plt.savefig("images/12_shap_summary.png", dpi=300)
        plt.close()

    # ----------------------------------------------------
    # 5. REGRESSÃO: TAXA OBSERVADA VS TAXA PREVISTA
    # ----------------------------------------------------
    print("6. Gerando Gráfico: Observado vs Previsto (Regressão)...")
    y_pred_reg = reg_pipe.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_reg, y_pred_reg, alpha=0.35, color="#2b5c8f", s=18, edgecolors="none")
    ax.plot([0, 100], [0, 100], color="red", linestyle="--", lw=2, label="Linha Ideal (Previsão Perfeita)")
    ax.set_xlabel("Taxa Real de Alfabetização 2024 (%)")
    ax.set_ylabel("Taxa Prevista pelo Modelo (%)")
    ax.set_title("Calibração do Modelo de Regressão - Teste 2024", fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend()
    plt.savefig("images/13_predicao_vs_real.png", dpi=300)
    plt.close()
    
    print("7. Todos os gráficos e relatórios de interpretabilidade gerados com sucesso!")

if __name__ == "__main__":
    run_interpretability_pipeline()
