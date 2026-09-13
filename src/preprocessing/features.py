# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

LEAKAGE_COLUMNS = [
    'taxa_alfabetizacao', 'nivel_alfabetizacao', 'gap_meta_2030', 'status_meta_2030',
    'classe_taxa', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025', 'meta_alfabetizacao_2026',
    'meta_alfabetizacao_2027', 'meta_alfabetizacao_2028', 'meta_alfabetizacao_2029', 'meta_alfabetizacao_2030',
    'computadores_por_aluno'
]

IDENTIFIER_COLUMNS = ['id_municipio', 'nome_municipio', 'sigla_uf', 'nome_uf']

def impute_macroeconomic_lag(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_2023 = df_out[df_out['ano'] == 2023].set_index('id_municipio')
    pib_2023_map = df_2023['pib'].to_dict()
    pop_2023_map = df_2023['populacao'].to_dict()

    mask_2024_missing = (df_out['ano'] == 2024) & (df_out['pib'].isnull())
    for idx in df_out[mask_2024_missing].index:
        m_id = df_out.loc[idx, 'id_municipio']
        pop_2024 = df_out.loc[idx, 'populacao']
        pib_2023 = pib_2023_map.get(m_id, np.nan)
        pop_2023 = pop_2023_map.get(m_id, np.nan)
        if pd.notnull(pib_2023) and pd.notnull(pop_2023) and pop_2023 > 0:
            pib_proj = pib_2023 * (pop_2024 / pop_2023)
            df_out.loc[idx, 'pib'] = pib_proj
            df_out.loc[idx, 'pib_per_capita'] = pib_proj / pop_2024 if pop_2024 > 0 else np.nan
    return df_out

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = impute_macroeconomic_lag(df)
    df_feat['log_populacao'] = np.log1p(df_feat['populacao'].clip(lower=0))
    df_feat['log_pib'] = np.log1p(df_feat['pib'].clip(lower=0))
    df_feat['log_pib_per_capita'] = np.log1p(df_feat['pib_per_capita'].clip(lower=0))

    qtd_escolas = df_feat['qtd_escolas'].replace(0, np.nan)
    df_feat['matriculas_por_escola'] = df_feat['matriculas_anos_iniciais'] / qtd_escolas
    df_feat['docentes_por_escola'] = df_feat['docentes_anos_iniciais'] / qtd_escolas

    df_feat['indice_infra_basica'] = (
        df_feat['pct_escolas_agua_potavel'].fillna(0) +
        df_feat['pct_escolas_energia_publica'].fillna(0) +
        df_feat['pct_escolas_esgoto_publico'].fillna(0)
    ) / 3.0

    df_feat['indice_recursos_pedagogicos'] = (
        df_feat['pct_escolas_internet'].fillna(0) +
        df_feat['pct_escolas_biblioteca'].fillna(0) +
        df_feat['pct_escolas_lab_informatica'].fillna(0) +
        df_feat['pct_escolas_quadra_esportes'].fillna(0)
    ) / 4.0

    df_feat['aluno_docente_critico'] = (df_feat['razao_aluno_docente'] > 22.0).astype(int)
    df_feat['esgoto_critico'] = (df_feat['pct_escolas_esgoto_publico'] < 0.20).astype(int)
    df_feat['biblioteca_adequada'] = (df_feat['pct_escolas_biblioteca'] >= 0.50).astype(int)

    df_feat['target_risco_educacional'] = (df_feat['taxa_alfabetizacao'] < 60.0).astype(int)
    df_feat['target_taxa_alfabetizacao'] = df_feat['taxa_alfabetizacao']
    return df_feat

def get_feature_names() -> Tuple[List[str], List[str]]:
    numeric_features = [
        'populacao', 'pib_per_capita', 'log_populacao', 'log_pib_per_capita',
        'qtd_escolas', 'matriculas_anos_iniciais', 'docentes_anos_iniciais',
        'razao_aluno_docente', 'matriculas_por_escola', 'docentes_por_escola',
        'pct_escolas_internet', 'pct_escolas_biblioteca', 'pct_escolas_lab_informatica',
        'pct_escolas_agua_potavel', 'pct_escolas_energia_publica', 'pct_escolas_esgoto_publico',
        'pct_escolas_quadra_esportes', 'indice_infra_basica', 'indice_recursos_pedagogicos',
        'aluno_docente_critico', 'esgoto_critico', 'biblioteca_adequada', 'percentual_participacao'
    ]
    categorical_features = ['nome_regiao', 'capital_uf', 'amazonia_legal']
    return numeric_features, categorical_features

def prepare_dataset(df: pd.DataFrame, target_type: str = 'classification') -> Tuple[pd.DataFrame, pd.Series]:
    df_proc = engineer_features(df)
    df_valid = df_proc.dropna(subset=['taxa_alfabetizacao']).copy()
    num_cols, cat_cols = get_feature_names()
    X = df_valid[num_cols + cat_cols].copy()
    if target_type == 'classification':
        y = df_valid['target_risco_educacional'].copy()
    else:
        y = df_valid['target_taxa_alfabetizacao'].copy()
    return X, y
