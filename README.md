# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil
### Pós-Graduação em Inteligência Artificial / AI Scientist

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Fase-3%20Conclu%C3%ADda-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositório contém a solução completa da etapa de **Análise Exploratória e Entendimento do Problema (EDA)** do Tech Challenge Fase 3, investigando a alfabetização infantil municipal no Brasil a partir dos dados da **Camada Gold**.

---

## 📁 Estrutura do Repositório

```text
tech-challenge-fase3/
│
├── data/
│   ├── raw/                      # Base íntegra da Camada Gold (Data.csv - 10.704 registros)
│   └── processed/                # Datasets limpos e tratados para modelagem
│
├── notebooks/
│   └── 01_analise_exploratoria.ipynb  # Notebook interativo executado de ponta a ponta
│
├── src/
│   ├── preprocessing/
│   │   └── data_loader.py        # Módulo de ingestão, tipagem e missing values
│   ├── visualization/
│   │   └── plots.py              # Visualizações em alta resolução (300 DPI)
│   └── eda_runner.py             # Script CLI que executa toda a pipeline analítica
│
├── reports/
│   └── relatorio_eda.md          # Relatório executivo detalhado de EDA e hipóteses
│
├── images/                       # Gráficos analíticos gerados pelo pipeline
│   ├── 01_distribuicao_taxa_alfabetizacao.png
│   ├── 02_comparativo_temporal_2023_2024.png
│   ├── 03_disparidade_regional_taxa.png
│   ├── 04_matriz_correlacao.png
│   ├── 05_infraestrutura_vs_alfabetizacao.png
│   ├── 06_razao_aluno_docente_vs_taxa.png
│   ├── 07_distribuicao_status_meta_2030.png
│   └── 08_amazonia_legal_vulnerabilidade.png
│
├── requirements.txt              # Dependências do projeto
├── README.md                     # Documentação oficial
└── .gitignore                    # Arquivos ignorados pelo Git
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar e Instalar Dependências
```bash
# Clone o repositório
git clone <url-do-repositorio>
cd tech-challenge-fase3

# Crie e ative um ambiente virtual
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Executar a Pipeline de EDA via Linha de Comando
```bash
python src/eda_runner.py
```
O script executa a validação das 36 variáveis, executa os testes estatísticos de hipóteses e regenera todos os gráficos de alta resolução na pasta `images/`.

### 3. Explorar o Notebook Jupyter
```bash
jupyter lab notebooks/01_analise_exploratoria.ipynb
```

---

## 📊 Principais Insights da Análise Exploratória

1. **Evolução Positiva Nacional**: A taxa média de alfabetização subiu de **60,48% em 2023 para 63,04% em 2024**, com **67,6% dos municípios brasileiros registrando avanço**.
2. **O Poder das Bibliotecas Escolares**: Municípios com cobertura $\ge 50\%$ de bibliotecas atingem média de **68,12%**, contra **61,18%** nos municípios sem cobertura satisfatória (diferença de $+6,94$ p.p., $p = 4,76 \times 10^{-34}$).
3. **O Gargalo do Saneamento na Amazônia Legal**: Municípios da Amazônia Legal sofrem com um déficit médio de **8,15 p.p.** na alfabetização, com apenas **7,7% das escolas conectadas à rede pública de esgoto** (contra 49,5% no restante do país).
4. **Sobrecarga Docente**: A razão aluno/docente apresenta correlação inversa significativa ($r_s = -0,15, p < 10^{-27}$), reforçando a necessidade de limites de alunos por turma no ciclo de alfabetização.
5. **O Benchmark do Ceará**: O Ceará lidera nacionalmente com **90,4% de taxa média em 2024**, demonstrando o impacto de incentivos fiscais atrelados a metas municipais e suporte pedagógico.

---

## 🛡️ Tratamento de Data Leakage e Preparação para Modelagem

Para a etapa supervisionada, foram isoladas e descartadas dos preditores $X$:
- `gap_meta_2030`: Relação determinística direta ($Taxa - Meta_{2030}$).
- `classe_taxa` e `nivel_alfabetizacao`: Discretizações diretas da taxa.
- `status_meta_2030`: Rótulo derivado do atingimento da meta de 2030.
- `meta_alfabetizacao_2025` a `2030`: Metas projetadas para anos futuros.

---

## 👥 Equipe
Projeto desenvolvido para o **Tech Challenge - Fase 3 (AI Scientist)**.
