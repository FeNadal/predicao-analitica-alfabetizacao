# -*- coding: utf-8 -*-
"""
Módulo de Inferência e Predição de Risco Educacional
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os
import sys
import joblib
import pandas as pd
from typing import Dict, Any, Optional

# Garante que a raiz do projeto esteja no sys.path e que os caminhos de
# modelo sejam absolutos, independente de onde o código é chamado
# (script, `python -m`, ou notebook em notebooks/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.preprocessing.data_loader import load_data
from src.preprocessing.features import engineer_features, get_feature_names

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_models():
    clf_path = os.path.join(PROCESSED_DIR, "best_classification_model.joblib")
    reg_path = os.path.join(PROCESSED_DIR, "best_regression_model.joblib")
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        raise FileNotFoundError(
            f"Modelos treinados não encontrados em {PROCESSED_DIR}. "
            "Execute 'python -m src.modeling.train' a partir da raiz do projeto primeiro."
        )
    clf_model = joblib.load(clf_path)
    reg_model = joblib.load(reg_path)
    return clf_model, reg_model


def predict_municipality(id_municipio: int, ano: int = 2024) -> Dict[str, Any]:
    df = load_data()
    df_feat = engineer_features(df)

    match = df_feat[(df_feat["id_municipio"] == id_municipio) & (df_feat["ano"] == ano)]
    if match.empty:
        raise ValueError(f"Município {id_municipio} não encontrado para o ano {ano}.")

    num_cols, cat_cols = get_feature_names()
    X = match[num_cols + cat_cols]

    clf_model, reg_model = load_models()

    pred_taxa = float(reg_model.predict(X)[0])
    pred_risco = int(clf_model.predict(X)[0])
    prob_risco = float(clf_model.predict_proba(X)[0, 1])

    taxa_real = float(match["taxa_alfabetizacao"].values[0]) if pd.notnull(match["taxa_alfabetizacao"].values[0]) else None

    return {
        "id_municipio": int(id_municipio),
        "nome_municipio": str(match["nome_municipio"].values[0]),
        "sigla_uf": str(match["sigla_uf"].values[0]),
        "regiao": str(match["nome_regiao"].values[0]),
        "taxa_alfabetizacao_real": taxa_real,
        "taxa_alfabetizacao_prevista": round(pred_taxa, 2),
        "risco_educacional_previsto": "Alto Risco" if pred_risco == 1 else "Baixo Risco",
        "probabilidade_alto_risco": round(prob_risco * 100, 2),
        "diagnostico": "Abaixo da meta de 60%" if pred_taxa < 60 else "No caminho da meta"
    }


if __name__ == "__main__":
    clf_m, reg_m = load_models()
    print("Modelos carregados com sucesso para inferência!")