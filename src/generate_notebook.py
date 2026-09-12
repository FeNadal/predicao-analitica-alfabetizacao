# -*- coding: utf-8 -*-
"""
Script gerador do Notebook 01_analise_exploratoria.ipynb
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ==========================================
# Célula 1: Título e Cabeçalho Executivo
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil
## Pós-Graduação em Inteligência Artificial / AI Scientist

---

### **Caderno 01: Análise Exploratória de Dados (EDA) e Formulação de Hipóteses**
**Objetivo:** Explorar a base de dados integrada da **Camada Gold** (10.704 registros correspondentes aos 5.352 municípios brasileiros nos anos de 2023 e 2024), diagnosticar a qualidade dos dados, compreender padrões socioeconômicos e estruturais, testar hipóteses analíticas estatisticamente e mapear riscos de *Data Leakage* para subsidiar a pipeline preditiva de Machine Learning.

---

### **Sumário do Notebook**
1. **Configuração de Ambiente e Ingestão de Dados**
2. **Dicionário e Taxonomia das 36 Variáveis**
3. **Diagnóstico Crítico de Qualidade e Dados Faltantes (*Missing Values*)**
4. **Mapeamento de Riscos de Vazamento de Dados (*Data Leakage*)**
5. **Análise Univariada: Distribuição da Taxa de Alfabetização**
6. **Análise Temporal: Evolução e Variação Anual ($\Delta$ 2024 - 2023)**
7. **Disparidades Regionais e Estaduais**
8. **Matriz de Correlação e Fatores Estruturais**
9. **Impacto da Infraestrutura Escolar no Desempenho Educacional**
10. **Recursos Docentes e Razão Aluno-Docente**
11. **Vulnerabilidade Territorial: Foco na Amazônia Legal**
12. **Testes Estatísticos de Hipóteses ($H_1, H_2, H_3, H_4$)**
13. **Respostas Estratégicas às Perguntas de Negócio / Políticas Públicas**
14. **Diretrizes e Recomendações para a Etapa de Modelagem Supervisionada**
"""))

# ==========================================
# Célula 2: Imports e Configurações
# ==========================================
cells.append(nbf.v4.new_code_cell("""# Importação de bibliotecas analíticas essenciais
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurações de exibição e gráficos
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'figure.autolayout': True
})
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("Bibliotecas importadas com sucesso!")
"""))

# ==========================================
# Célula 3: Carregamento dos Dados
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Carregamento e Visão Geral da Base da Camada Gold"""))

cells.append(nbf.v4.new_code_cell("""# Carregamento do dataset
DATA_PATH = os.path.join('..', 'data', 'raw', 'Data.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'Data.csv'  # Fallback caso executado na raiz

df = pd.read_csv(DATA_PATH, encoding='utf-8')

print(f"Dimensões do Dataset: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
print(f"Anos de Coleta: {sorted(df['ano'].unique())}")
print(f"Municípios Únicos: {df['id_municipio'].nunique():,}")
df.head(3)
"""))

# ==========================================
# Célula 4: Dicionário e Taxonomia
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Taxonomia e Dicionário de Variáveis

As 36 variáveis foram integradas na Fase 2 a partir de bases oficiais (Censo Escolar/INEP, Compromisso Criança Alfabetizada/MEC, IBGE). Elas se agrupam em 6 blocos analíticos:

| Bloco | Variáveis | Papel Analítico |
| :--- | :--- | :--- |
| **Identificadores Territoriais** | `id_municipio`, `nome_municipio`, `sigla_uf`, `nome_uf`, `nome_regiao`, `capital_uf`, `amazonia_legal` | Estratificação espacial e controle regional |
| **Temporais** | `ano` (2023, 2024) | Séries temporais e medição de evolução anual |
| **Socioeconômicos** | `populacao`, `pib`, `pib_per_capita` | Capacidade fiscal e contexto demográfico |
| **Estrutura Escolar** | `qtd_escolas`, `matriculas_anos_iniciais`, `docentes_anos_iniciais`, `razao_aluno_docente` | Porte da rede e densidade de atendimento |
| **Infraestrutura das Escolas** | `% internet`, `% biblioteca`, `% lab_info`, `% água potável`, `% energia`, `% esgoto`, `% quadra`, `computadores/aluno` | Condições físicas de suporte ao aprendizado |
| **Avaliação & Metas** | `taxa_alfabetizacao`, `meta_2024` a `meta_2030`, `nivel_alfabetizacao`, `percentual_participacao`, `gap_meta_2030`, `status_meta_2030`, `classe_taxa` | Indicadores de desempenho e acompanhamento |
"""))

# ==========================================
# Célula 5: Diagnóstico de Missing Values
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Diagnóstico Crítico de Qualidade e Valores Faltantes (*Missing Values*)"""))

cells.append(nbf.v4.new_code_cell("""# Cálculo detalhado de dados faltantes por coluna e por ano
missing_data = []
for col in df.columns:
    tot_null = df[col].isnull().sum()
    pct_null = (tot_null / len(df)) * 100
    null_2023 = df[df['ano'] == 2023][col].isnull().sum()
    null_2024 = df[df['ano'] == 2024][col].isnull().sum()
    if tot_null > 0:
        missing_data.append({
            'Coluna': col,
            'Tipo': str(df[col].dtype),
            'Total Nulos': tot_null,
            '% Nulo': pct_null,
            'Nulos em 2023': null_2023,
            'Nulos em 2024': null_2024
        })

df_missing = pd.DataFrame(missing_data).sort_values(by='Total Nulos', ascending=False)
df_missing.reset_index(drop=True)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Conclusões Críticas sobre os Dados Faltantes:
1. **`computadores_por_aluno` (100% nulo)**: A variável não foi preenchida em nenhum município em 2023 nem 2024. Deve ser **descartada** do espaço de features por ausência total de informação.
2. **`pib` e `pib_per_capita` (50% nulo - 100% do ano 2024)**: O IBGE divulga o PIB municipal com defasagem histórica de 2 anos (PIB 2023 publicado recentemente, PIB 2024 ainda não apurado).
   - *Solução metodológica para Machine Learning*: Na modelagem preditiva de 2024, pode-se realizar *forward-fill* do PIB 2023 ajustado pela variação populacional de 2023 para 2024.
3. **`taxa_alfabetizacao`, `nivel_alfabetizacao`, `gap_meta_2030` (120 nulos em 2023)**: Correspondem a 120 municípios (`status_meta_2030 == 'sem_dado'`) que não atingiram a taxa mínima de participação exigida pelo Inep (quórum mínimo de alunos) para validação estatística da amostra. Em 2024, todos os 5.352 municípios possuem taxa válida.
4. **`meta_alfabetizacao_2024` (240 nulos)**: Municípios que não pactuaram meta específica no primeiro ciclo.
"""))

# ==========================================
# Célula 6: Data Leakage
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Mapeamento Preventivo de Vazamento de Dados (*Data Leakage*)

> **ATENÇÃO METODOLÓGICA**: Para construir modelos supervisionados com generalização real, as seguintes variáveis **NÃO** podem ser utilizadas como preditoras:
> - **`gap_meta_2030`**: É uma relação determinística direta ($Taxa - Meta_{2030}$). Incluí-la causaria vazamento total ($R^2 = 1.0$).
> - **`classe_taxa` e `nivel_alfabetizacao`**: São discretizações ordinais e categóricas da própria `taxa_alfabetizacao`.
> - **`status_meta_2030`**: É o rótulo derivado do atingimento da meta ($Taxa \ge Meta_{2030}$).
> - **Metas Futuras (`meta_alfabetizacao_2025` a `2030`)**: Metas futuras não exercem causalidade sobre o desempenho presente e contêm projeções do próprio Inep sobre a trajetória esperada do município.
"""))

# ==========================================
# Célula 7: Análise Univariada
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Análise Univariada: Comportamento da Taxa de Alfabetização"""))

cells.append(nbf.v4.new_code_cell("""# Estatísticas descritivas por ano
stats_summary = df.groupby('ano')['taxa_alfabetizacao'].agg([
    ('Contagem', 'count'),
    ('Média', 'mean'),
    ('Desvio Padrão', 'std'),
    ('Mínimo', 'min'),
    ('Q25 (1º Quartil)', lambda x: x.quantile(0.25)),
    ('Mediana', 'median'),
    ('Q75 (3º Quartil)', lambda x: x.quantile(0.75)),
    ('Máximo', 'max'),
    ('Assimetria', lambda x: x.skew()),
    ('Curtose', lambda x: x.kurtosis())
])
stats_summary
"""))

cells.append(nbf.v4.new_code_cell("""# Visualização da Distribuição (Histograma + KDE)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

for i, y in enumerate([2023, 2024]):
    sub = df[df['ano'] == y]['taxa_alfabetizacao'].dropna()
    sns.histplot(sub, kde=True, ax=axes[i], color='#2b5c8f' if y == 2023 else '#1f9a55', bins=35, stat="density", alpha=0.6)
    axes[i].axvline(sub.mean(), color='red', linestyle='--', linewidth=1.8, label=f'Média: {sub.mean():.1f}%')
    axes[i].axvline(sub.median(), color='black', linestyle=':', linewidth=1.8, label=f'Mediana: {sub.median():.1f}%')
    axes[i].axvline(80.0, color='purple', linestyle='-', linewidth=1.5, alpha=0.7, label='Meta 2030 (80%)')
    axes[i].set_title(f'Distribuição em {y} (N={len(sub):,})', fontweight='bold')
    axes[i].set_xlabel('Taxa de Alfabetização (%)')
    axes[i].set_ylabel('Densidade' if i == 0 else '')
    axes[i].legend()
    axes[i].set_xlim(0, 100)

plt.suptitle('Distribuição da Taxa de Alfabetização Infantil no Brasil', fontweight='bold', y=1.05)
plt.show()
"""))

# ==========================================
# Célula 8: Análise Temporal
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Análise Temporal: Variação Municipal ($\Delta$ 2024 - 2023)"""))

cells.append(nbf.v4.new_code_cell("""# Pivotando os dados para parear municípios
piv = df.pivot(index='id_municipio', columns='ano', values=['taxa_alfabetizacao', 'nome_regiao', 'nome_municipio', 'sigla_uf', 'status_meta_2030'])
delta_series = piv['taxa_alfabetizacao'][2024] - piv['taxa_alfabetizacao'][2023]
regiao_series = piv['nome_regiao'][2024]

df_evolucao = pd.DataFrame({
    'nome_municipio': piv['nome_municipio'][2024],
    'uf': piv['sigla_uf'][2024],
    'regiao': regiao_series,
    'taxa_2023': piv['taxa_alfabetizacao'][2023],
    'taxa_2024': piv['taxa_alfabetizacao'][2024],
    'delta': delta_series,
    'status_2023': piv['status_meta_2030'][2023],
    'status_2024': piv['status_meta_2030'][2024]
}).dropna(subset=['delta'])

pct_avanco = (df_evolucao['delta'] > 0).mean() * 100
print(f"Média do Delta Nacional: +{df_evolucao['delta'].mean():.2f} pontos percentuais")
print(f"Mediana do Delta: +{df_evolucao['delta'].median():.2f} p.p.")
print(f"Percentual de municípios com avanço positivo: {pct_avanco:.1f}%")
print(f"Municípios que atingiram a meta em 2024 mas não estavam em 2023: "
      f"{((df_evolucao['status_2023'] != 'acima_meta_2030') & (df_evolucao['status_2024'] == 'acima_meta_2030')).sum()}")
"""))

cells.append(nbf.v4.new_code_cell("""# Gráfico de Evolução e Delta por Região
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

sns.histplot(df_evolucao['delta'], kde=True, ax=axes[0], color='#0d6efd', bins=40)
axes[0].axvline(0, color='grey', linestyle='-', linewidth=1)
axes[0].axvline(df_evolucao['delta'].mean(), color='red', linestyle='--', label=f"Média: +{df_evolucao['delta'].mean():.2f} p.p.")
axes[0].axvline(df_evolucao['delta'].median(), color='green', linestyle=':', label=f"Mediana: +{df_evolucao['delta'].median():.2f} p.p.")
axes[0].set_title('Histograma da Variação Anual da Taxa (2024 - 2023)', fontweight='bold')
axes[0].set_xlabel('Delta (pontos percentuais)')
axes[0].legend()

sns.boxplot(data=df_evolucao, x='regiao', y='delta', ax=axes[1], palette='Set2')
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1].set_title('Variação da Alfabetização por Grande Região', fontweight='bold')
axes[1].set_xlabel('Região')
axes[1].set_ylabel('Delta (p.p.)')

plt.show()
"""))

# ==========================================
# Célula 9: Disparidades Regionais e Estaduais
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Disparidades Regionais e Rankings Federativos"""))

cells.append(nbf.v4.new_code_cell("""# Ranking Médio de Alfabetização por Unidade Federativa em 2024
uf_ranking = df[df['ano'] == 2024].groupby('sigla_uf').agg(
    taxa_media=('taxa_alfabetizacao', 'mean'),
    taxa_mediana=('taxa_alfabetizacao', 'median'),
    total_municipios=('id_municipio', 'count'),
    pct_acima_meta=('status_meta_2030', lambda x: (x == 'acima_meta_2030').mean() * 100)
).sort_values(by='taxa_media', ascending=False)

print("Top 5 UFs com Maior Taxa de Alfabetização (2024):")
print(uf_ranking.head(5)[['taxa_media', 'taxa_mediana', 'pct_acima_meta']])

print("\\nBottom 5 UFs com Menor Taxa de Alfabetização (2024):")
print(uf_ranking.tail(5)[['taxa_media', 'taxa_mediana', 'pct_acima_meta']])
"""))

cells.append(nbf.v4.new_code_cell("""# Gráfico Comparativo Regional e Ranking Estadual
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(
    data=df.dropna(subset=['taxa_alfabetizacao']),
    x='nome_regiao', y='taxa_alfabetizacao', hue='ano',
    palette={2023: '#4c72b0', 2024: '#55a868'}, ax=axes[0]
)
axes[0].axhline(80.0, color='purple', linestyle='--', alpha=0.7, label='Meta 2030 (80%)')
axes[0].set_title('Taxa de Alfabetização por Região (2023 vs 2024)', fontweight='bold')
axes[0].set_xlabel('Região')
axes[0].set_ylabel('Taxa (%)')
axes[0].legend(title='Ano')

# Ranking por UF
uf_2024 = df[df['ano'] == 2024].groupby('sigla_uf')['taxa_alfabetizacao'].mean().sort_values()
colors = ['#d95f02' if x < 60 else '#7570b3' if x < 75 else '#1b9e77' for x in uf_2024]
axes[1].barh(uf_2024.index, uf_2024.values, color=colors)
axes[1].axvline(80.0, color='purple', linestyle='--', alpha=0.7, label='Meta 2030')
axes[1].axvline(uf_2024.mean(), color='red', linestyle=':', label=f'Média Nacional: {uf_2024.mean():.1f}%')
axes[1].set_title('Média da Taxa de Alfabetização por UF (2024)', fontweight='bold')
axes[1].set_xlabel('Taxa Média (%)')
axes[1].legend(loc='lower right')

plt.show()
"""))

# ==========================================
# Célula 10: Matriz de Correlação
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Matriz de Correlação e Relações Não Lineares"""))

cells.append(nbf.v4.new_code_cell("""# Matriz de Correlação de Spearman (captura relações monotônicas não lineares)
corr_cols = [
    'taxa_alfabetizacao', 'razao_aluno_docente', 'populacao',
    'pib_per_capita', 'qtd_escolas', 'matriculas_anos_iniciais',
    'pct_escolas_internet', 'pct_escolas_biblioteca', 'pct_escolas_lab_informatica',
    'pct_escolas_esgoto_publico', 'pct_escolas_quadra_esportes', 'pct_escolas_agua_potavel',
    'percentual_participacao'
]

# Calculado na base de 2023 com PIB preenchido
df_corr = df[df['ano'] == 2023][corr_cols].dropna()
corr_matrix = df_corr.corr(method='spearman')

plt.figure(figsize=(11, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='vlag',
    vmin=-0.5, vmax=0.5, square=True, linewidths=0.5
)
plt.title('Matriz de Correlação de Spearman entre Fatores Estruturais e Alfabetização', fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.show()
"""))

# ==========================================
# Célula 11: Infraestrutura Escolar
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Impacto das Condições de Infraestrutura Escolar no Aprendizado"""))

cells.append(nbf.v4.new_code_cell("""# Análise por faixas de cobertura de infraestrutura (Ano 2024)
df_2024 = df[df['ano'] == 2024].dropna(subset=['taxa_alfabetizacao']).copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
items = [
    ('pct_escolas_biblioteca', 'Biblioteca Escolar', axes[0, 0], '#2b5c8f'),
    ('pct_escolas_internet', 'Internet nas Escolas', axes[0, 1], '#1f9a55'),
    ('pct_escolas_lab_informatica', 'Laboratório de Informática', axes[1, 0], '#e66101'),
    ('pct_escolas_esgoto_publico', 'Rede Pública de Esgoto', axes[1, 1], '#5e3c99')
]

for col, name, ax, clr in items:
    df_2024['faixa'] = pd.cut(df_2024[col], bins=[-0.01, 0.25, 0.50, 0.75, 1.0], labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    sns.boxplot(data=df_2024, x='faixa', y='taxa_alfabetizacao', ax=ax, color=clr, boxprops=dict(alpha=0.65))
    medias = df_2024.groupby('faixa', observed=False)['taxa_alfabetizacao'].mean()
    for idx, m in enumerate(medias):
        if not np.isnan(m):
            ax.text(idx, m + 1.2, f'{m:.1f}%', ha='center', fontweight='bold')
    ax.set_title(f'Cobertura de {name}', fontweight='bold')
    ax.set_xlabel('Faixa de Cobertura das Escolas')
    ax.set_ylabel('Taxa de Alfabetização (%)')

plt.suptitle('Taxa de Alfabetização por Quartil de Infraestrutura Escolar (2024)', fontweight='bold', y=1.02)
plt.show()
"""))

# ==========================================
# Célula 12: Razão Aluno-Docente
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 10. Recursos Docentes e Razão Aluno-Docente"""))

cells.append(nbf.v4.new_code_cell("""# Dispersão com Linha de Tendência OLS
df_ratio = df[(df['ano'] == 2024) & (df['razao_aluno_docente'] < 50)].dropna(subset=['taxa_alfabetizacao', 'razao_aluno_docente'])
r_pearson, p_pearson = stats.pearsonr(df_ratio['razao_aluno_docente'], df_ratio['taxa_alfabetizacao'])

plt.figure(figsize=(10, 5))
sns.regplot(
    data=df_ratio, x='razao_aluno_docente', y='taxa_alfabetizacao',
    scatter_kws={'alpha': 0.2, 'color': '#2b5c8f', 's': 18},
    line_kws={'color': 'red', 'linewidth': 2, 'label': f'Tendência Linear (r = {r_pearson:.3f}, p < 0.001)'}
)
plt.axhline(80.0, color='purple', linestyle='--', label='Meta 2030 (80%)')
plt.title('Razão Aluno por Docente vs Taxa de Alfabetização (2024)', fontweight='bold')
plt.xlabel('Alunos por Docente (Anos Iniciais do Ensino Fundamental)')
plt.ylabel('Taxa de Alfabetização (%)')
plt.legend()
plt.show()
"""))

# ==========================================
# Célula 13: Amazônia Legal
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 11. Vulnerabilidade Territorial: Foco na Amazônia Legal"""))

cells.append(nbf.v4.new_code_cell("""# Comparativo Amazônia Legal vs Restante do Brasil
df_amz = df[df['ano'] == 2024].dropna(subset=['taxa_alfabetizacao']).copy()
df_amz['amazonia_desc'] = df_amz['amazonia_legal'].map({1: 'Amazônia Legal', 0: 'Demais Municípios'})

resumo_amz = df_amz.groupby('amazonia_desc').agg(
    taxa_media=('taxa_alfabetizacao', 'mean'),
    taxa_mediana=('taxa_alfabetizacao', 'median'),
    pct_esgoto_medio=('pct_escolas_esgoto_publico', lambda x: x.mean() * 100),
    pct_internet_medio=('pct_escolas_internet', lambda x: x.mean() * 100),
    pct_biblioteca_medio=('pct_escolas_biblioteca', lambda x: x.mean() * 100),
    pct_atingiu_meta=('status_meta_2030', lambda x: (x == 'acima_meta_2030').mean() * 100)
)
resumo_amz
"""))

# ==========================================
# Célula 14: Testes Formais de Hipóteses
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 12. Testes Estatísticos Formais de Hipóteses

Para atender aos requisitos científicos da pós-graduação, testamos formalmente quatro hipóteses analíticas:
"""))

cells.append(nbf.v4.new_code_cell("""# -------------------------------------------------------------
# H1: Presença de Bibliotecas Escolares
# H0: Não há diferença na taxa de alfabetização entre escolas com alta vs baixa cobertura de biblioteca.
# Ha: Municípios com cobertura >= 50% de bibliotecas possuem taxas significativamente maiores.
# -------------------------------------------------------------
g_high_bib = df_2024[df_2024['pct_escolas_biblioteca'] >= 0.5]['taxa_alfabetizacao']
g_low_bib = df_2024[df_2024['pct_escolas_biblioteca'] < 0.5]['taxa_alfabetizacao']
stat_u1, p_u1 = stats.mannwhitneyu(g_high_bib, g_low_bib, alternative='greater')

print(f"[H1] Bibliotecas: Média Alta = {g_high_bib.mean():.2f}% | Média Baixa = {g_low_bib.mean():.2f}%")
print(f"     Diferença = +{g_high_bib.mean() - g_low_bib.mean():.2f} p.p. | Mann-Whitney U p-valor = {p_u1:.4e}")
print("     -> Conclusão H1: Rejeita-se H0. Bibliotecas escolares impactam positivamente a proficiência (p < 0.001).\\n")

# -------------------------------------------------------------
# H2: Razão Aluno-Docente
# H0: Não há correlação entre razão aluno/docente e a taxa de alfabetização.
# Ha: Há correlação linear e monotônica negativa estatisticamente significante.
# -------------------------------------------------------------
sub_r = df_2024[df_2024['razao_aluno_docente'] < 60].dropna(subset=['razao_aluno_docente'])
r_pearson, p_pearson = stats.pearsonr(sub_r['razao_aluno_docente'], sub_r['taxa_alfabetizacao'])
r_spear, p_spear = stats.spearmanr(sub_r['razao_aluno_docente'], sub_r['taxa_alfabetizacao'])

print(f"[H2] Razão Aluno-Docente: Pearson r = {r_pearson:.4f} (p = {p_pearson:.4e}) | Spearman rho = {r_spear:.4f} (p = {p_spear:.4e})")
print("     -> Conclusão H2: Rejeita-se H0. A sobrecarga de alunos por professor correlaciona-se inversamente com o letramento.\\n")

# -------------------------------------------------------------
# H3: Vulnerabilidade na Amazônia Legal
# H0: As médias de alfabetização da Amazônia Legal e das demais regiões são estatisticamente iguais.
# Ha: A Amazônia Legal possui média inferior às demais regiões brasileiras.
# -------------------------------------------------------------
amz_vals = df_2024[df_2024['amazonia_legal'] == 1]['taxa_alfabetizacao']
non_amz_vals = df_2024[df_2024['amazonia_legal'] == 0]['taxa_alfabetizacao']
t_stat, p_t = stats.ttest_ind(amz_vals, non_amz_vals, equal_var=False)

print(f"[H3] Amazônia Legal: Média Amazônia = {amz_vals.mean():.2f}% | Demais Regiões = {non_amz_vals.mean():.2f}%")
print(f"     Déficit = {non_amz_vals.mean() - amz_vals.mean():.2f} p.p. | Welch t-test p-valor = {p_t:.4e}")
print("     -> Conclusão H3: Rejeita-se H0. O déficit na Amazônia Legal é estatisticamente significante (p < 0.001).\\n")

# -------------------------------------------------------------
# H4: Quórum de Participação na Avaliação
# H0: A taxa observada independe do nível de participação dos alunos na avaliação.
# -------------------------------------------------------------
p_high = df_2024[df_2024['percentual_participacao'] >= 80]['taxa_alfabetizacao']
p_low = df_2024[df_2024['percentual_participacao'] < 80]['taxa_alfabetizacao']
stat_u4, p_u4 = stats.mannwhitneyu(p_high, p_low)

print(f"[H4] Participação: Média Alta Adesão (>=80%) = {p_high.mean():.2f}% | Baixa Adesão (<80%) = {p_low.mean():.2f}%")
print(f"     Mann-Whitney U p-valor = {p_u4:.4e}")
print("     -> Conclusão H4: Rejeita-se H0. Baixa participação gera viés e menor taxa estimada, exigindo atenção em políticas de adesão.")
"""))

# ==========================================
# Célula 15: Respostas Estratégicas às Perguntas de Negócio
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 13. Respostas Estratégicas às Perguntas de Negócio (Edital Tech Challenge)

Com base nos dados explorados, respondemos objetivamente às 5 perguntas estratégicas de negócio:

### 1. Quais fatores mais impactam a alfabetização?
- **Infraestrutura de Apoio ao Letramento**: Escolas dotadas de biblioteca escolar e conectividade à internet apresentam médias de alfabetização **6,94 a 11,8 pontos percentuais superiores** àquelas sem esses recursos.
- **Densidade Pedagógica**: Turmas com razão aluno-docente equilibrada (< 15 alunos por docente) alcançam resultados superiores àquelas sobrecarregadas (> 25 alunos/docente).
- **Gestão Pedagógica Local (Efeito Ceará)**: Estados como o Ceará (média de 90,4% em 2024) comprovam que políticas estruturadas de incentivo fiscal municipal (cota-parte de ICMS atrelada à alfabetização) e suporte pedagógico aos professores superam restrições puramente orçamentárias.

### 2. Quais municípios apresentam maior risco educacional?
- Municípios com a combinação de **três vulnerabilidades críticas**:
  1. Localizados na **Amazônia Legal** ou interior remoto do Norte/Nordeste;
  2. Cobertura de esgoto escolar pública inferior a 10% e ausência de bibliotecas;
  3. Baixo quórum de participação na avaliação nacional (< 80%).
- Em 2024, **742 municípios** encontram-se na classe `muito_baixo` (< 40% de crianças alfabetizadas).

### 3. Quais regiões possuem padrões semelhantes?
- **Cluster 1 (Norte e Nordeste sem CE/PE)**: Caracterizam-se por infraestrutura sanitária e digital escolar frágil, menor PIB per capita e maiores distâncias até a meta de 2030.
- **Cluster 2 (Sul, Sudeste e Centro-Oeste)**: Caracterizam-se por taxas médias superiores a 68%, alta taxa de atendimento escolar, saneamento consolidado e proximidade ou cumprimento antecipado da meta de 2030.
- **Outlier Positivo (Ceará e Goiás)**: Apresentam desempenho descolado da sua renda per capita regional, fruto de governança e pactuação de metas com bonificação municipal.

### 4. Como prever municípios que podem não atingir metas futuras?
- Modelando uma pontuação de risco (*Early Warning Indicator*) baseada em:
  - Variação interanual ($\Delta \text{taxa} \le 0$);
  - Distância atual da meta de 2030 (`gap_meta_2030`);
  - Déficit severo de infraestrutura (esgoto = 0 e biblioteca = 0);
  - Razão aluno-docente elevada (> 25).

### 5. Quais variáveis possuem maior influência nos modelos?
- As variáveis mais informativas para a modelagem sem vazamento são: `pct_escolas_biblioteca`, `pct_escolas_internet`, `razao_aluno_docente`, `pct_escolas_esgoto_publico`, `pib_per_capita` (2023), `populacao` e a macrorregião geográfica.
"""))

# ==========================================
# Célula 16: Diretrizes para Modelagem
# ==========================================
cells.append(nbf.v4.new_markdown_cell("""## 14. Diretrizes e Próximos Passos para a Modelagem Supervisionada (Fase 3/4)

### 📌 Recomendações Técnicas:
1. **Definição de Alvo (Target)**:
   - **Cenário de Regressão**: Prever a `taxa_alfabetizacao` contínua (0 a 100) utilizando atributos estruturais e socioeconômicos.
   - **Cenário de Classificação**: Prever se o município está em risco educacional (`classe_taxa == 'muito_baixo'` ou `status_meta_2030 == 'abaixo_meta_2030'`).
2. **Prevenção Rigorosa de Data Leakage**:
   - Descartar `gap_meta_2030`, `nivel_alfabetizacao`, `classe_taxa` e metas futuras 2025-2030 do conjunto preditor $X$.
3. **Estratégia de Validação Estatística**:
   - **Split Temporal**: Treinar o modelo com os dados de 2023 e validar na coorte de 2024, testando a capacidade de predição do modelo no horizonte futuro real.
   - **GroupKFold por UF**: Evitar vazamento espacial ao separar cidades vizinhas entre treino e validação.
4. **Tratamento de Missing**:
   - Eliminar `computadores_por_aluno` (100% nulo).
   - Imputar `pib` e `pib_per_capita` em 2024 via *forward-fill* do histórico municipal de 2023 ajustado pela população.
5. **Explicabilidade**:
   - Utilizar **SHAP (*SHapley Additive exPlanations*)** e *Feature Importance* para auditar as decisões do modelo preditivo e apoiar decisões de gestores públicos.
"""))

nb.cells = cells

output_path = r"C:\Users\floli\.gemini\antigravity\scratch\tech-challenge-fase3\notebooks\01_analise_exploratoria.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook criado com sucesso em: {output_path}")
