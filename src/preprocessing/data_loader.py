# -*- coding: utf-8 -*-
"""
Módulo de Carregamento e Diagnóstico de Dados da Camada Gold
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'raw', 'Data.csv'
)


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Carrega o dataset da Camada Gold com tipagem e encoding adequados."""
    path = file_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f'Arquivo não encontrado em: {path}')

    df = pd.read_csv(path, encoding='utf-8')
    return df


def get_variable_classification() -> Dict[str, List[str]]:
    """Classifica as variáveis da base em grupos analíticos e identifica riscos de leakage."""
    return {
        'identificadores_territoriais': [
            'id_municipio', 'nome_municipio', 'sigla_uf', 'nome_uf', 
            'nome_regiao', 'capital_uf', 'amazonia_legal'
        ],
        'temporais': ['ano'],
        'socioeconomicos': ['populacao', 'pib', 'pib_per_capita'],
        'estrutura_escolar': [
            'qtd_escolas', 'matriculas_anos_iniciais', 'docentes_anos_iniciais',
            'razao_aluno_docente'
        ],
        'infraestrutura_escolas': [
            'pct_escolas_internet', 'pct_escolas_biblioteca', 'pct_escolas_lab_informatica',
            'pct_escolas_agua_potavel', 'pct_escolas_energia_publica', 
            'pct_escolas_esgoto_publico', 'pct_escolas_quadra_esportes',
            'computadores_por_aluno'
        ],
        'avaliacoes_metas': [
            'taxa_alfabetizacao', 'meta_alfabetizacao_2024', 'meta_alfabetizacao_2025',
            'meta_alfabetizacao_2026', 'meta_alfabetizacao_2027', 'meta_alfabetizacao_2028',
            'meta_alfabetizacao_2029', 'meta_alfabetizacao_2030', 'nivel_alfabetizacao',
            'percentual_participacao', 'gap_meta_2030', 'status_meta_2030', 'classe_taxa'
        ],
        'risco_data_leakage': [
            'gap_meta_2030',           # Derivado direto: taxa - meta_2030
            'classe_taxa',             # Categorização direta da taxa
            'nivel_alfabetizacao',     # Discretização direta da taxa
            'status_meta_2030',        # Rótulo alvo (se o objetivo for prever a taxa contínua)
            'meta_alfabetizacao_2025', # Metas futuras posteriores ao ano observado
            'meta_alfabetizacao_2026',
            'meta_alfabetizacao_2027',
            'meta_alfabetizacao_2028',
            'meta_alfabetizacao_2029',
            'meta_alfabetizacao_2030'
        ]
    }


def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Gera relatório completo de dados faltantes por coluna e por ano."""
    records = []
    for col in df.columns:
        total_missing = df[col].isnull().sum()
        pct_missing = (total_missing / len(df)) * 100
        missing_2023 = df[df['ano'] == 2023][col].isnull().sum()
        missing_2024 = df[df['ano'] == 2024][col].isnull().sum()
        dtype = str(df[col].dtype)

        records.append({
            'coluna': col,
            'tipo': dtype,
            'total_missing': total_missing,
            'pct_missing': round(pct_missing, 2),
            'missing_2023': missing_2023,
            'missing_2024': missing_2024
        })

    report = pd.DataFrame(records)
    return report.sort_values(by='total_missing', ascending=False)
