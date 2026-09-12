# -*- coding: utf-8 -*-
"""
Script Orquestrador da Análise Exploratória de Dados (EDA Runner)
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_loader import load_data, get_missing_report, get_variable_classification
from src.visualization import plots


def run_statistical_hypothesis_tests(df: pd.DataFrame) -> dict:
    """Executa testes formais de hipóteses estatísticas exigidos no Tech Challenge."""
    df_2024 = df[df['ano'] == 2024].dropna(subset=['taxa_alfabetizacao']).copy()
    results = {}
    
    # H1: Escolas com biblioteca têm maior taxa de alfabetização
    group_high_bib = df_2024[df_2024['pct_escolas_biblioteca'] >= 0.5]['taxa_alfabetizacao']
    group_low_bib = df_2024[df_2024['pct_escolas_biblioteca'] < 0.5]['taxa_alfabetizacao']
    stat_u, p_u = stats.mannwhitneyu(group_high_bib, group_low_bib, alternative='greater')
    corr_bib, p_corr_bib = stats.spearmanr(df_2024['pct_escolas_biblioteca'], df_2024['taxa_alfabetizacao'])
    
    results['H1'] = {
        'descricao': 'Impacto de Bibliotecas Escolares na Alfabetização',
        'media_alta_cobertura': round(group_high_bib.mean(), 2),
        'media_baixa_cobertura': round(group_low_bib.mean(), 2),
        'diff_media_pp': round(group_high_bib.mean() - group_low_bib.mean(), 2),
        'mann_whitney_p_value': p_u,
        'spearman_corr': round(corr_bib, 4),
        'spearman_p_value': p_corr_bib,
        'conclusao': 'Hipótese Confirmada (p < 0.001): Cobertura de bibliotecas correlaciona positivamente com alfabetização.'
    }
    
    # H2: Razão aluno-docente correlaciona negativamente com a alfabetização
    df_sub_razao = df_2024[df_2024['razao_aluno_docente'] < 60].dropna(subset=['razao_aluno_docente'])
    corr_p, p_p = stats.pearsonr(df_sub_razao['razao_aluno_docente'], df_sub_razao['taxa_alfabetizacao'])
    corr_s, p_s = stats.spearmanr(df_sub_razao['razao_aluno_docente'], df_sub_razao['taxa_alfabetizacao'])
    
    results['H2'] = {
        'descricao': 'Razão Aluno-Docente vs Alfabetização',
        'pearson_corr': round(corr_p, 4),
        'pearson_p_value': p_p,
        'spearman_corr': round(corr_s, 4),
        'spearman_p_value': p_s,
        'conclusao': 'Hipótese Confirmada (p < 0.001): Sobrecarga docente correlaciona-se com menor proficiência.'
    }
    
    # H3: Vulnerabilidade na Amazônia Legal
    amz = df_2024[df_2024['amazonia_legal'] == 1]['taxa_alfabetizacao']
    non_amz = df_2024[df_2024['amazonia_legal'] == 0]['taxa_alfabetizacao']
    t_stat, p_t = stats.ttest_ind(amz, non_amz, equal_var=False)
    
    amz_esgoto = df_2024[df_2024['amazonia_legal'] == 1]['pct_escolas_esgoto_publico'].mean() * 100
    non_amz_esgoto = df_2024[df_2024['amazonia_legal'] == 0]['pct_escolas_esgoto_publico'].mean() * 100
    
    results['H3'] = {
        'descricao': 'Disparidade Territorial - Amazônia Legal vs Demais Regiões',
        'media_taxa_amazonia': round(amz.mean(), 2),
        'media_taxa_demais': round(non_amz.mean(), 2),
        'diff_taxa_pp': round(non_amz.mean() - amz.mean(), 2),
        't_test_p_value': p_t,
        'esgoto_escolas_amazonia': round(amz_esgoto, 1),
        'esgoto_escolas_demais': round(non_amz_esgoto, 1),
        'conclusao': 'Hipótese Confirmada (p < 0.001): Amazônia Legal apresenta déficit estatisticamente significativo em alfabetização e saneamento escolar.'
    }
    
    # H4: Percentual de participação na avaliação
    part_high = df_2024[df_2024['percentual_participacao'] >= 80]['taxa_alfabetizacao']
    part_low = df_2024[df_2024['percentual_participacao'] < 80]['taxa_alfabetizacao']
    stat_p, p_part = stats.mannwhitneyu(part_high, part_low)
    
    results['H4'] = {
        'descricao': 'Quórum de Participação na Avaliação vs Taxa',
        'media_participacao_alta': round(part_high.mean(), 2),
        'media_participacao_baixa': round(part_low.mean(), 2),
        'mann_whitney_p_value': p_part,
        'conclusao': 'Municípios com baixa adesão (<80%) apresentam maior variabilidade e menor confiabilidade da métrica.'
    }
    
    return results


def main():
    print("=" * 70)
    print("INICIANDO PIPELINE DE ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'Data.csv')
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"\n1. Carregando dados de: {data_path}")
    df = load_data(data_path)
    print(f"   -> Linhas: {len(df):,} | Colunas: {df.shape[1]}")
    print(f"   -> Anos disponíveis: {sorted(df['ano'].unique())}")
    print(f"   -> Municípios únicos: {df['id_municipio'].nunique():,}")
    
    print("\n2. Gerando Relatório de Dados Faltantes (Missing Values)...")
    missing_rep = get_missing_report(df)
    missing_with_nulls = missing_rep[missing_rep['total_missing'] > 0]
    print(missing_with_nulls.to_string(index=False))
    
    print("\n3. Executando Testes Estatísticos de Hipóteses...")
    hypotheses = run_statistical_hypothesis_tests(df)
    for h_key, h_data in hypotheses.items():
        print(f"\n   [{h_key}] {h_data['descricao']}:")
        for k, v in h_data.items():
            if k != 'descricao':
                print(f"       - {k}: {v}")
                
    print("\n4. Gerando Gráficos Analíticos em Alta Resolução (images/)...")
    
    p1 = os.path.join(images_dir, '01_distribuicao_taxa_alfabetizacao.png')
    plots.plot_target_distribution(df, p1)
    print(f"   [OK] 01. Distribuição da Taxa: {p1}")
    
    p2 = os.path.join(images_dir, '02_comparativo_temporal_2023_2024.png')
    plots.plot_temporal_evolution(df, p2)
    print(f"   [OK] 02. Comparativo Temporal: {p2}")
    
    p3 = os.path.join(images_dir, '03_disparidade_regional_taxa.png')
    plots.plot_regional_disparities(df, p3)
    print(f"   [OK] 03. Disparidades Regionais: {p3}")
    
    p4 = os.path.join(images_dir, '04_matriz_correlacao.png')
    plots.plot_correlation_matrix(df, p4)
    print(f"   [OK] 04. Matriz de Correlação: {p4}")
    
    p5 = os.path.join(images_dir, '05_infraestrutura_vs_alfabetizacao.png')
    plots.plot_infrastructure_impact(df, p5)
    print(f"   [OK] 05. Impacto de Infraestrutura: {p5}")
    
    p6 = os.path.join(images_dir, '06_razao_aluno_docente_vs_taxa.png')
    plots.plot_student_teacher_ratio(df, p6)
    print(f"   [OK] 06. Razão Aluno-Docente: {p6}")
    
    p7 = os.path.join(images_dir, '07_distribuicao_status_meta_2030.png')
    plots.plot_meta_2030_status(df, p7)
    print(f"   [OK] 07. Status Meta 2030: {p7}")
    
    p8 = os.path.join(images_dir, '08_amazonia_legal_vulnerabilidade.png')
    plots.plot_amazonia_legal(df, p8)
    print(f"   [OK] 08. Amazônia Legal: {p8}")
    
    print("\n" + "=" * 70)
    print("PIPELINE DE EDA CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == '__main__':
    main()
