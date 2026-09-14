# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil
### Pós-Graduação em Inteligência Artificial / AI Scientist

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Fase-3%20Conclu%C3%ADda-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Contexto do Problema

A alfabetização até o 2º ano do Ensino Fundamental é um dos indicadores mais sensíveis do
desenvolvimento educacional brasileiro. Apesar de metas nacionais e estaduais bem definidas,
municípios apresentam disparidades expressivas de desempenho, muitas vezes associadas a fatores
territoriais, socioeconômicos e de infraestrutura escolar que não são óbvios à primeira vista.

Este projeto usa a **camada Gold** construída no Tech Challenge da Fase 2 — que integra o
Indicador Criança Alfabetizada, metas nacionais/estaduais/municipais, dados territoriais,
socioeconômicos, educacionais complementares e populacionais — para investigar esses padrões e
construir modelos preditivos capazes de apoiar decisões de política pública.

## 🎯 Objetivo Analítico

1. Entender, via Análise Exploratória de Dados (EDA), quais fatores se associam à taxa de
   alfabetização municipal.
2. Construir e validar modelos supervisionados de **classificação** (município em risco
   educacional ou não) e **regressão** (taxa de alfabetização prevista).
3. Interpretar os modelos (Feature Importance e SHAP) para identificar as variáveis mais
   relevantes e traduzir isso em recomendações acionáveis para gestores públicos.

## 📊 Descrição da Base

- **Fonte**: camada Gold produzida no Tech Challenge Fase 2, disponível em
  [jadeferreira/tech-challenge-alfabetizacao](https://github.com/jadeferreira/tech-challenge-alfabetizacao).
- **Volume**: `Data.csv` com 10.704 registros, cobrindo os anos de **2023 e 2024**.
- **Granularidade**: um registro por município/ano.
- **Grupos de variáveis**: identificadores territoriais, dados temporais, socioeconômicos
  (população, PIB, PIB per capita), estrutura escolar (nº de escolas, matrículas, docentes,
  razão aluno/docente), infraestrutura escolar (internet, biblioteca, laboratório, saneamento,
  energia, quadra), e indicadores de avaliação/metas (taxa de alfabetização, metas 2024–2030,
  nível de alfabetização, status da meta 2030).

> ⚠️ A pasta `data/` está no `.gitignore` e **não é versionada** neste repositório (arquivo
> pesado). Para rodar o projeto, baixe o `Data.csv` a partir do repositório da Fase 2 linkado
> acima e salve em `data/raw/Data.csv`.

## 📁 Estrutura do Repositório

```text
tech-challenge-fase3/
│
├── data/
│   ├── raw/                          # Data.csv (camada Gold, não versionado)
│   └── processed/                    # Modelos treinados e predições geradas pela pipeline
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb       # EDA completa, testes de hipótese
│   └── 02_modelagem_supervisionada.ipynb   # Pipeline, treino, avaliação e SHAP
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py            # Ingestão, tipagem, relatório de missing values
│   │   ├── features.py               # Engenharia de atributos e colunas de leakage
│   │   └── pipeline.py               # ColumnTransformer + Pipeline Scikit-learn
│   ├── modeling/
│   │   ├── train.py                  # Treino, validação cruzada e seleção do melhor modelo
│   │   └── predict.py                # Inferência para um município específico
│   ├── evaluation/
│   │   ├── metrics.py                # Métricas de classificação e regressão
│   │   └── interpretability.py       # Matriz de confusão, ROC, Feature Importance, SHAP
│   ├── visualization/
│   │   └── plots.py                  # Gráficos da EDA em alta resolução (300 DPI)
│   └── eda_runner.py                 # Script CLI que executa a pipeline analítica de EDA
│
├── reports/
│   └── relatorio_eda.md              # Relatório executivo de EDA e hipóteses
│
├── images/                           # Gráficos gerados pela pipeline
│   ├── 01–08_*.png                   # EDA (distribuições, correlações, disparidades regionais)
│   ├── 09_matriz_confusao.png        # Avaliação do classificador
│   ├── 10_curva_roc.png              # Avaliação do classificador
│   ├── 11_feature_importance.png     # Interpretabilidade
│   ├── 12_shap_summary.png           # Interpretabilidade (SHAP)
│   └── 13_predicao_vs_real.png       # Avaliação do regressor
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar e instalar dependências
```bash
git clone <url-do-repositorio>
cd tech-challenge-fase3

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 2. Obter a base de dados
Baixe o `Data.csv` do repositório da Fase 2
([jadeferreira/tech-challenge-alfabetizacao](https://github.com/jadeferreira/tech-challenge-alfabetizacao))
e salve em `data/raw/Data.csv`.

### 3. Rodar a Análise Exploratória (EDA)
```bash
python src/eda_runner.py
```
Valida as variáveis, roda os testes estatísticos de hipóteses e gera os gráficos `01`–`08` em
`images/`.

### 4. Treinar e validar os modelos supervisionados
```bash
python -m src.modeling.train
```
Executa a engenharia de atributos, separa o treino (2023) do teste (2024) de forma temporal
(evitando vazamento entre anos), roda validação cruzada de 5 folds, compara 3 classificadores e
3 regressores, e salva:
- `data/processed/best_classification_model.joblib`
- `data/processed/best_regression_model.joblib`
- `data/processed/test_predictions_2024.csv`
- `data/processed/evaluation_summary.json` (métricas completas de todos os modelos testados)

### 5. Gerar interpretabilidade e gráficos de avaliação
```bash
python -m src.evaluation.interpretability
```
Gera matriz de confusão, curva ROC, Feature Importance e o gráfico SHAP (`images/09`–`13`), além
de `data/processed/top_features.json`.

### 6. Predizer o risco de um município específico
```bash
python -m src.modeling.predict
```

### 7. Explorar os notebooks interativamente
```bash
jupyter lab notebooks/01_analise_exploratoria.ipynb
jupyter lab notebooks/02_modelagem_supervisionada.ipynb
```

---

## 🧪 Etapas de Modelagem

1. **Engenharia de atributos** (`features.py`): índices compostos de infraestrutura básica e de
   recursos pedagógicos, log-transformações de população/PIB, imputação de PIB 2024 ausente via
   projeção pela variação populacional 2023→2024, e flags binárias de situação crítica
   (razão aluno/docente, saneamento, biblioteca).
2. **Prevenção de data leakage**: variáveis derivadas diretamente da taxa de alfabetização ou de
   metas futuras (`gap_meta_2030`, `classe_taxa`, `nivel_alfabetizacao`, `status_meta_2030`,
   `meta_alfabetizacao_2025` a `2030`, `computadores_por_aluno`) são isoladas antes de entrar no
   conjunto de preditores.
3. **Split temporal**: treino com os dados de **2023** e teste com os dados de **2024** — em vez
   de um split aleatório — para simular o cenário real de previsão de um ano futuro.
4. **Pipeline Scikit-learn integrada** (`pipeline.py`): `SimpleImputer` (mediana para
   numéricas, moda para categóricas) → `RobustScaler` / `OneHotEncoder` → modelo, tudo dentro de
   um único `Pipeline`/`ColumnTransformer`, garantindo que o pré-processamento seja aprendido
   apenas nos dados de treino.
5. **Validação cruzada + teste hold-out**: `StratifiedKFold` (classificação) e `KFold`
   (regressão) com 5 folds sobre o treino, seguidos de avaliação final no conjunto de teste de
   2024.

## ⚙️ Escolha do Algoritmo

Foram comparados três modelos para cada tarefa, e o melhor de cada grupo foi selecionado
automaticamente pela métrica de teste (ROC-AUC para classificação, R² para regressão):

| Tarefa | Modelos comparados |
|---|---|
| Classificação (risco educacional: taxa < 60%) | Regressão Logística (baseline, `class_weight="balanced"`), Random Forest Classifier, HistGradientBoosting Classifier |
| Regressão (taxa de alfabetização prevista) | Ridge Regression (baseline), Random Forest Regressor, HistGradientBoosting Regressor |

O modelo vencedor de cada tarefa é salvo automaticamente em `data/processed/*.joblib`.

## 📈 Métricas de Avaliação

- **Classificação**: acurácia, acurácia balanceada, precisão, recall, F1-score, ROC-AUC e
  matriz de confusão.
- **Regressão**: MAE, MSE, RMSE, R² e MAPE.

Os valores exatos de cada modelo (validação cruzada e teste) ficam registrados em
`data/processed/evaluation_summary.json` após rodar `src/modeling/train.py` — atualize esta
seção do README com os números do seu run mais recente antes da entrega final.

## 🔍 Interpretação dos Resultados

A interpretabilidade é feita em duas camadas, geradas por `src/evaluation/interpretability.py`:
- **Feature Importance** nativa do modelo vencedor (`images/11_feature_importance.png`).
- **SHAP Values** (`images/12_shap_summary.png`), calculados sobre uma amostra do conjunto de
  teste, mostrando direção e magnitude do impacto de cada variável na predição de risco.

*(Preencha aqui, após rodar a pipeline, quais variáveis apareceram no topo do ranking e como
isso se conecta aos achados da EDA — ex.: infraestrutura escolar, razão aluno/docente, região.)*

## 💡 Principais Insights da Análise Exploratória

1. **Evolução Positiva Nacional**: a taxa média de alfabetização subiu de **60,48% em 2023**
   para **63,04% em 2024**, com **67,6% dos municípios** registrando avanço.
2. **O Poder das Bibliotecas Escolares**: municípios com cobertura ≥ 50% de bibliotecas atingem
   média de **68,12%**, contra **61,18%** nos demais (diferença de +6,94 p.p., p ≈ 4,76×10⁻³⁴).
3. **O Gargalo do Saneamento na Amazônia Legal**: déficit médio de **8,15 p.p.** na
   alfabetização, com apenas **7,7%** das escolas conectadas à rede pública de esgoto (contra
   49,5% no restante do país).
4. **Sobrecarga Docente**: correlação inversa significativa entre razão aluno/docente e taxa de
   alfabetização (rₛ = −0,15, p < 10⁻²⁷).
5. **O Benchmark do Ceará**: lidera nacionalmente com **90,4%** de taxa média em 2024.

## ⚠️ Limitações do Projeto

- A granularidade é municipal, não individual — os modelos apoiam priorização de políticas por
  município, não avaliam ou rotulam alunos ou escolas específicas.
- A base cobre apenas dois anos (2023–2024), o que limita a robustez de uma validação temporal
  mais longa.
- Associações encontradas via correlação e Feature Importance/SHAP indicam relação preditiva,
  não causalidade.
- Dados socioeconômicos regionais (PIB) parcialmente ausentes em 2024 foram estimados por
  projeção simples a partir de 2023, o que introduz incerteza adicional nesses casos.

## 🏛️ Aplicação Prática para Políticas Públicas

Os modelos e o ranking de importância de variáveis permitem:
- Priorizar investimento em infraestrutura escolar (saneamento, bibliotecas) nos municípios e
  regiões com maior déficit relativo, como a Amazônia Legal.
- Sinalizar antecipadamente municípios em risco de não atingir a meta de alfabetização,
  apoiando a alocação de recursos e suporte pedagógico.
- Usar `src/modeling/predict.py` para gerar diagnósticos pontuais por município, como insumo
  para reuniões com gestores e equipes pedagógicas.

## 🔮 Possíveis Evoluções Futuras

- Incorporar novas fontes externas (Censo Escolar, FUNDEB, Atlas do Desenvolvimento Humano,
  PNAD, Cadastro Único) para enriquecer o conjunto de preditores.
- Testar modelos hierárquicos que respeitem a estrutura aninhada (aluno/escola/município).
- Ampliar a janela temporal assim que novos anos de dados estiverem disponíveis, permitindo
  validação temporal mais robusta.
- Empacotar a inferência (`predict.py`) como uma API ou dashboard para uso direto por gestores.

## 🛡️ Tratamento de Data Leakage

Colunas isoladas e descartadas do conjunto de preditores $X$:
- `gap_meta_2030`: relação determinística direta ($Taxa - Meta_{2030}$).
- `classe_taxa` e `nivel_alfabetizacao`: discretizações diretas da taxa.
- `status_meta_2030`: rótulo derivado do atingimento da meta de 2030.
- `meta_alfabetizacao_2025` a `2030`: metas projetadas para anos futuros.
- `computadores_por_aluno`: excluída por risco de vazamento identificado na análise da base.

---

## 👥 Equipe

Projeto desenvolvido para o **Tech Challenge - Fase 3 (AI Scientist)**.
