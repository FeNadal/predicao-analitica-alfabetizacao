# -*- coding: utf-8 -*-
"""
Módulo de Visualização de Dados - Gráficos Analíticos de Alta Resolução
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configuração global de estilo
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 15
})


def plot_target_distribution(df: pd.DataFrame, output_path: str):
    """01. Distribuição da Taxa de Alfabetização Infantil (2023 vs 2024)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    colors = {2023: '#2b5c8f', 2024: '#1f9a55'}
    
    for i, year in enumerate([2023, 2024]):
        sub = df[df['ano'] == year]['taxa_alfabetizacao'].dropna()
        mean_val = sub.mean()
        median_val = sub.median()
        
        sns.histplot(sub, kde=True, ax=axes[i], color=colors[year], bins=35, stat="density", alpha=0.6)
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.8, label=f'Média: {mean_val:.1f}%')
        axes[i].axvline(median_val, color='black', linestyle=':', linewidth=1.8, label=f'Mediana: {median_val:.1f}%')
        axes[i].axvline(80.0, color='purple', linestyle='-', linewidth=1.5, alpha=0.7, label='Meta 2030 (80%)')
        
        axes[i].set_title(f'Ano {year} (N = {len(sub):,} municípios)', fontweight='bold')
        axes[i].set_xlabel('Taxa de Alfabetização (%)')
        axes[i].set_ylabel('Densidade' if i == 0 else '')
        axes[i].legend(loc='upper left')
        axes[i].set_xlim(0, 100)
    
    fig.suptitle('Distribuição da Taxa de Alfabetização Municipal no Brasil', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temporal_evolution(df: pd.DataFrame, output_path: str):
    """02. Comparativo Temporal e Evolução Municipal (Delta 2024 - 2023)."""
    pivoted = df.pivot(index='id_municipio', columns='ano', values=['taxa_alfabetizacao', 'nome_regiao', 'nome_municipio', 'sigla_uf'])
    taxas = pivoted['taxa_alfabetizacao'].dropna()
    taxas['delta'] = taxas[2024] - taxas[2023]
    taxas['regiao'] = pivoted['nome_regiao'][2024]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histograma do Delta
    mean_delta = taxas['delta'].mean()
    median_delta = taxas['delta'].median()
    positive_share = (taxas['delta'] > 0).mean() * 100
    
    sns.histplot(taxas['delta'], kde=True, ax=axes[0], color='#0d6efd', bins=40)
    axes[0].axvline(0, color='grey', linestyle='-', linewidth=1)
    axes[0].axvline(mean_delta, color='red', linestyle='--', label=f'Média de Evolução: +{mean_delta:.2f} p.p.')
    axes[0].axvline(median_delta, color='green', linestyle=':', label=f'Mediana: +{median_delta:.2f} p.p.')
    axes[0].set_title(f'Variação Anual da Taxa ({positive_share:.1f}% dos municípios avançaram)', fontweight='bold')
    axes[0].set_xlabel('Delta (Taxa 2024 - Taxa 2023 em p.p.)')
    axes[0].set_ylabel('Quantidade de Municípios')
    axes[0].legend(loc='upper left')
    
    # Delta por Região
    sns.boxplot(data=taxas, x='regiao', y='delta', ax=axes[1], palette='Set2')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('Evolução da Taxa (Delta) por Grande Região', fontweight='bold')
    axes[1].set_xlabel('Região')
    axes[1].set_ylabel('Variação (p.p.)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_regional_disparities(df: pd.DataFrame, output_path: str):
    """03. Disparidade Regional e Rankings Estaduais (2024)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot por Região
    sns.boxplot(
        data=df.dropna(subset=['taxa_alfabetizacao']),
        x='nome_regiao',
        y='taxa_alfabetizacao',
        hue='ano',
        palette={2023: '#4c72b0', 2024: '#55a868'},
        ax=axes[0]
    )
    axes[0].axhline(80.0, color='purple', linestyle='--', alpha=0.7, label='Meta Nacional 2030 (80%)')
    axes[0].set_title('Taxa de Alfabetização por Região Geográfica (2023 vs 2024)', fontweight='bold')
    axes[0].set_xlabel('Grande Região')
    axes[0].set_ylabel('Taxa de Alfabetização (%)')
    axes[0].legend(title='Ano', loc='lower right')
    
    # Ranking Médio por UF em 2024
    uf_2024 = df[df['ano'] == 2024].groupby('sigla_uf')['taxa_alfabetizacao'].mean().sort_values()
    colors = ['#d95f02' if x < 60 else '#7570b3' if x < 75 else '#1b9e77' for x in uf_2024]
    
    axes[1].barh(uf_2024.index, uf_2024.values, color=colors)
    axes[1].axvline(80.0, color='purple', linestyle='--', alpha=0.7, label='Meta 2030 (80%)')
    axes[1].axvline(uf_2024.mean(), color='red', linestyle=':', label=f'Média Brasil: {uf_2024.mean():.1f}%')
    axes[1].set_title('Média da Taxa de Alfabetização por Unidade Federativa (2024)', fontweight='bold')
    axes[1].set_xlabel('Taxa Média de Alfabetização (%)')
    axes[1].set_ylabel('UF')
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_path: str):
    """04. Matriz de Correlação Linear e Não Linear das Variáveis Estruturais."""
    features = [
        'taxa_alfabetizacao', 'razao_aluno_docente', 'populacao',
        'pib_per_capita', 'qtd_escolas', 'matriculas_anos_iniciais',
        'pct_escolas_internet', 'pct_escolas_biblioteca', 'pct_escolas_lab_informatica',
        'pct_escolas_esgoto_publico', 'pct_escolas_quadra_esportes', 'pct_escolas_agua_potavel',
        'percentual_participacao'
    ]
    
    # Usar 2023 onde o PIB está preenchido
    df_2023 = df[df['ano'] == 2023][features].dropna()
    corr = df_2023.corr(method='spearman')
    
    # Nomes mais legíveis
    labels = [
        'Taxa Alfabetização', 'Razão Aluno/Docente', 'População',
        'PIB per Capita', 'Qtd Escolas', 'Matrículas',
        '% Internet', '% Biblioteca', '% Lab Informática',
        '% Esgoto Público', '% Quadra Esportes', '% Água Potável',
        '% Participação Avaliação'
    ]
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap='vlag',
        vmin=-0.6, vmax=0.6, square=True, linewidths=0.5,
        xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.8}
    )
    plt.title('Matriz de Correlação de Spearman (Ano Base 2023 com PIB Integrado)', fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_infrastructure_impact(df: pd.DataFrame, output_path: str):
    """05. Impacto da Infraestrutura Escolar na Alfabetização."""
    df_clean = df[df['ano'] == 2024].dropna(subset=['taxa_alfabetizacao']).copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    infra_vars = [
        ('pct_escolas_biblioteca', 'Biblioteca Escolar', axes[0, 0], '#2b5c8f'),
        ('pct_escolas_internet', 'Internet nas Escolas', axes[0, 1], '#1f9a55'),
        ('pct_escolas_lab_informatica', 'Laboratório de Informática', axes[1, 0], '#e66101'),
        ('pct_escolas_esgoto_publico', 'Rede Pública de Esgoto', axes[1, 1], '#5e3c99')
    ]
    
    for var, title, ax, color in infra_vars:
        # Categorizar em faixas (0-25%, 25-50%, 50-75%, 75-100%)
        df_clean['faixa'] = pd.cut(
            df_clean[var],
            bins=[-0.01, 0.25, 0.50, 0.75, 1.0],
            labels=['0-25%', '25-50%', '50-75%', '75-100%']
        )
        
        sns.boxplot(data=df_clean, x='faixa', y='taxa_alfabetizacao', ax=ax, color=color, boxprops=dict(alpha=0.7))
        mean_by_faixa = df_clean.groupby('faixa', observed=False)['taxa_alfabetizacao'].mean()
        
        for idx, mean_v in enumerate(mean_by_faixa):
            if not np.isnan(mean_v):
                ax.text(idx, mean_v + 1.5, f'{mean_v:.1f}%', horizontalalignment='center', fontweight='bold', color='black')
        
        ax.set_title(f'Cobertura de {title}', fontweight='bold')
        ax.set_xlabel('Faixa de Cobertura das Escolas')
        ax.set_ylabel('Taxa de Alfabetização (%)')
    
    fig.suptitle('Taxa de Alfabetização por Faixas de Infraestrutura Escolar (2024)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_student_teacher_ratio(df: pd.DataFrame, output_path: str):
    """06. Razão Aluno-Docente vs Taxa de Alfabetização e Porte Escolar."""
    df_sub = df[(df['ano'] == 2024) & (df['razao_aluno_docente'] < 50)].dropna(subset=['taxa_alfabetizacao', 'razao_aluno_docente']).copy()
    
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df_sub, x='razao_aluno_docente', y='taxa_alfabetizacao',
        scatter_kws={'alpha': 0.25, 'color': '#2b5c8f', 's': 20},
        line_kws={'color': 'red', 'linewidth': 2, 'label': 'Tendência Linear (OLS)'}
    )
    plt.axhline(80.0, color='purple', linestyle='--', label='Meta 2030 (80%)')
    
    corr = df_sub['razao_aluno_docente'].corr(df_sub['taxa_alfabetizacao'])
    plt.title(f'Razão Aluno-Docente vs Taxa de Alfabetização (2024) | r = {corr:.3f}', fontweight='bold')
    plt.xlabel('Razão Aluno por Docente (Anos Iniciais)')
    plt.ylabel('Taxa de Alfabetização (%)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_meta_2030_status(df: pd.DataFrame, output_path: str):
    """07. Status Frente à Meta de 2030 por Região (2023 vs 2024)."""
    df_valid = df[df['status_meta_2030'].isin(['abaixo_meta_2030', 'acima_meta_2030'])].copy()
    
    counts = df_valid.groupby(['ano', 'nome_regiao', 'status_meta_2030']).size().unstack(fill_value=0)
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for i, year in enumerate([2023, 2024]):
        sub_pct = pcts.loc[year]
        sub_pct.plot(
            kind='bar', stacked=True, ax=axes[i],
            color={'abaixo_meta_2030': '#d95f02', 'acima_meta_2030': '#1b9e77'}
        )
        axes[i].set_title(f'Situação em {year}', fontweight='bold')
        axes[i].set_xlabel('Região')
        axes[i].set_ylabel('% de Municípios' if i == 0 else '')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].axhline(50, color='white', linestyle=':', alpha=0.8)
        axes[i].legend(['Abaixo da Meta 2030', 'Acima da Meta 2030'], loc='upper right')
    
    fig.suptitle('Proporção de Municípios Acima e Abaixo da Meta de 2030 por Região', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_amazonia_legal(df: pd.DataFrame, output_path: str):
    """08. Vulnerabilidade Educacional e Infraestrutural na Amazônia Legal."""
    df_2024 = df[df['ano'] == 2024].dropna(subset=['taxa_alfabetizacao']).copy()
    df_2024['amazonia_desc'] = df_2024['amazonia_legal'].map({1: 'Amazônia Legal', 0: 'Demais Municípios'})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Alfabetização
    sns.boxplot(data=df_2024, x='amazonia_desc', y='taxa_alfabetizacao', ax=axes[0], palette=['#2b83ba', '#d7191c'])
    axes[0].set_title('Taxa de Alfabetização (2024)', fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Taxa (%)')
    
    # Comparativo de Infraestrutura Média
    infra_cols = ['pct_escolas_esgoto_publico', 'pct_escolas_internet', 'pct_escolas_biblioteca', 'pct_escolas_lab_informatica']
    labels = ['Esgoto Público', 'Internet', 'Biblioteca', 'Lab. Info']
    means = df_2024.groupby('amazonia_desc')[infra_cols].mean() * 100
    
    means.T.plot(kind='bar', ax=axes[1], color=['#2b83ba', '#d7191c'], width=0.7)
    axes[1].set_title('Acesso Médio à Infraestrutura Escolar (%)', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('% Escolas com Acesso')
    axes[1].set_xticklabels(labels, rotation=30, ha='right')
    axes[1].legend(title='')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
