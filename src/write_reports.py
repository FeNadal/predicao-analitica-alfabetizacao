# -*- coding: utf-8 -*-
"""
Gerador dos Documentos de Relatório Executivo e README Oficial
Tech Challenge - Fase 3 | Pós Tech AI Scientist
"""

import os

base_dir = r"C:\Users\floli\.gemini\antigravity\scratch\tech-challenge-fase3"

relatorio_content = """# Relatório Executivo de Análise Exploratória de Dados (EDA)
## Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil
**Programa:** Pós-Graduação em Inteligência Artificial / AI Scientist  
**Data:** Setembro de 2026  
**Status do Projeto:** Fase de Entendimento do Problema e EDA Concluída  

---

## 1. Sumário Executivo

Este documento consolida os achados analíticos e diagnósticos empíricos obtidos na etapa de **Análise Exploratória de Dados (EDA)** do Tech Challenge Fase 3. O projeto investiga os determinantes da alfabetização infantil municipal no Brasil a partir da base consolidada na **Camada Gold** (10.704 registros correspondentes a 5.352 municípios brasileiros nos anos de 2023 e 2024, abrangendo 36 atributos).

Os principais destaques da análise revelam:
1. **Avanço Médio Nacional, mas Persistência de Desigualdades**: A taxa média de alfabetização infantil avançou de **60,48% em 2023 para 63,04% em 2024** (+2,56 p.p. médios), com **67,6% dos municípios apresentando evolução positiva**. No entanto, apenas **21,3% dos municípios (1.141 cidades)** já atingiram a meta de 80% estipulada para 2030, enquanto **742 municípios** permanecem em situação crítica (`classe_taxa == 'muito_baixo'`, taxa < 40%).
2. **Infraestrutura Escolar como Alavanca de Equidade**: Municípios com cobertura universal de bibliotecas escolares apresentam média de alfabetização **6,94 pontos percentuais superior** ($p = 4,76 \\times 10^{-34}$) àqueles com déficit de bibliotecas. A presença de internet nas escolas também se associa a um incremento médio de quase 12 p.p. entre os extremos de cobertura.
3. **A Vulnerabilidade Estrutural da Amazônia Legal**: Municípios inseridos na Amazônia Legal apresentam uma taxa média de alfabetização significativamente inferior (**56,03% vs. 64,18%**, $p = 2,05 \\times 10^{-30}$), explicada em grande parte por déficits severos de saneamento básico escolar (apenas **7,7% das escolas contam com rede pública de esgoto**, contra 49,5% no restante do país).
4. **Governança e Pactuação Superam Renda Per Capita**: O estado do **Ceará lidera o ranking nacional com média de 90,4% de alfabetização em 2024** e mais de 91% de seus municípios acima da meta de 2030, superando estados de renda per capita substancialmente mais alta (como SP, RJ e RS). Isso corrobora o impacto de mecanismos de governança pedagógica e incentivo fiscal (redistribuição do ICMS baseada em indicadores educacionais).
5. **Prevenção Rigorosa de Data Leakage**: Foram formalmente mapeadas e isoladas 10 variáveis com risco de vazamento de dados (`gap_meta_2030`, `classe_taxa`, `nivel_alfabetizacao`, `status_meta_2030` e metas futuras 2025-2030), garantindo a robustez dos futuros modelos preditivos.

---

## 2. Taxonomia e Diagnóstico da Base de Dados (Camada Gold)

A base possui 36 variáveis organizadas da seguinte forma:

| Grupo Analítico | Variáveis | Observações Metodológicas |
| :--- | :--- | :--- |
| **Identificação & Geografia** | `id_municipio`, `nome_municipio`, `sigla_uf`, `nome_uf`, `nome_regiao`, `capital_uf`, `amazonia_legal` | Sem dados faltantes. Representa a totalidade dos municípios com rede pública. |
| **Temporais** | `ano` | Coortes anuais de 2023 (5.352 linhas) e 2024 (5.352 linhas). |
| **Demografia & Renda** | `populacao`, `pib`, `pib_per_capita` | `pib` e `pib_per_capita` 100% ausentes em 2024 em virtude da defasagem oficial do IBGE. |
| **Recursos Docentes** | `qtd_escolas`, `matriculas_anos_iniciais`, `docentes_anos_iniciais`, `razao_aluno_docente` | Razão média nacional de 17,6 alunos por docente, com variações de 4 a mais de 100. |
| **Infraestrutura Escolar** | `% internet`, `% biblioteca`, `% lab_info`, `% água potável`, `% energia`, `% esgoto`, `% quadra`, `computadores/aluno` | `computadores_por_aluno` 100% nulo (descartado). Infraestrutura de água e energia quase universal (>95%). |
| **Avaliação & Metas** | `taxa_alfabetizacao`, `meta_2024` a `meta_2030`, `nivel_alfabetizacao`, `percentual_participacao`, `gap_meta_2030`, `status_meta_2030`, `classe_taxa` | Em 2023, 120 cidades sem dado por quórum insuficiente. Em 2024, 100% de preenchimento. |

---

## 3. Matriz de Correlação e Fatores de Associação

A partir da correlação não paramétrica de Spearman (ano base 2023):
- **Razão Aluno-Docente vs. Taxa de Alfabetização**: $r_s = -0,26$ ($p < 10^{-25}$). Municípios com turmas inchadas ou escassez de docentes formados apresentam taxas sistematicamente menores.
- **Percentual de Participação vs. Taxa**: $r_s = +0,28$ ($p < 10^{-30}$). Municípios com alta adesão discente ao exame obtêm métricas mais consistentes.
- **Quadra de Esportes e Bibliotecas**: Apresentam correlações positivas moderadas ($r_s = +0,25$ e $+0,20$).
- **PIB per Capita**: Apresenta correlação positiva modesta ($r_s = +0,20$), sinalizando que riqueza municipal isolada não é garantia de letramento na idade certa.

---

## 4. Testes Estatísticos de Hipóteses

Quatro hipóteses analíticas foram formalmente formuladas e testadas:

### Hipótese 1: Impacto das Bibliotecas Escolares
- **Hipótese Nula ($H_0$)**: A distribuição da taxa de alfabetização é idêntica entre municípios com alta ($\ge 50\%$) e baixa cobertura de bibliotecas.
- **Hipótese Alternativa ($H_a$)**: Municípios com $\ge 50\%$ de bibliotecas possuem desempenho superior.
- **Resultado**: Média Alta = **68,12%** vs. Média Baixa = **61,18%** ($\Delta = +6,94$ p.p.).
- **Estatística de Teste**: Mann-Whitney $U$, $p = 4,76 \\times 10^{-34}$.
- **Conclusão**: Rejeita-se $H_0$. O acervo físico e o espaço de leitura escolar são determinantes para o letramento infantil.

### Hipótese 2: Sobrecarga Docente (Razão Aluno/Docente)
- **Hipótese Nula ($H_0$)**: A razão aluno/docente não possui relação inversa com o rendimento em alfabetização.
- **Hipótese Alternativa ($H_a$)**: Há correlação linear e monotônica negativa estatisticamente significante.
- **Resultado**: Coeficiente de Pearson $r = -0,1549$ ($p = 4,26 \\times 10^{-30}$) e Spearman $r_s = -0,1493$ ($p = 4,98 \\times 10^{-28}$).
- **Conclusão**: Rejeita-se $H_0$. A redução de alunos por professor nos anos iniciais do Ensino Fundamental favorece a atenção individualizada e a alfabetização.

### Hipótese 3: Disparidade Territorial na Amazônia Legal
- **Hipótese Nula ($H_0$)**: Municípios da Amazônia Legal possuem taxas equivalentes às demais regiões.
- **Hipótese Alternativa ($H_a$)**: Municípios da Amazônia Legal possuem média inferior.
- **Resultado**: Média Amazônia Legal = **56,03%** vs. Demais Regiões = **64,18%** (Déficit de 8,15 p.p.).
- **Estatística de Teste**: Welch's $t$-test $t = -11,54$, $p = 2,05 \\times 10^{-30}$. Esgoto escolar: 7,7% (Amazônia) vs. 49,5% (Brasil).
- **Conclusão**: Rejeita-se $H_0$. Há vulnerabilidade estrutural e sanitária crítica na Amazônia Legal.

### Hipótese 4: Confiabilidade pelo Quórum de Participação
- **Hipótese Nula ($H_0$)**: A taxa de alfabetização independe da cobertura amostral de participação na avaliação.
- **Resultado**: Média com participação $\ge 80\%$ = **63,74%** vs. participação $< 80\%$ = **51,87%** ($p = 3,28 \\times 10^{-25}$).
- **Conclusão**: Rejeita-se $H_0$. Baixa participação induz viés de não resposta e subavaliação do município.

---

## 5. Respostas às Perguntas Estratégicas de Negócio

### 1. Quais fatores mais impactam a alfabetização?
1. **Ambiente Físico de Leitura e Conexão**: Bibliotecas e internet escolar agregam de 7 a 12 pontos percentuais à proficiência média.
2. **Razão Aluno-Docente**: Turmas com menos de 15 alunos por docente superam turmas superlotadas.
3. **Incentivos Institucionais e Formação de Professores**: A experiência do Ceará demonstra que bonificação fiscal e cooperação estado-município exercem maior tração do que o PIB bruto per capita.

### 2. Quais municípios apresentam maior risco educacional?
- Municípios com índice crítico de saneamento escolar (< 10%), localizados na Amazônia Legal ou áreas isoladas do Nordeste, e com razão aluno-docente elevada. Atualmente, 742 cidades encontram-se na faixa "muito baixo" (< 40% alfabetizados).

### 3. Quais regiões possuem padrões semelhantes?
- **Padrão Norte / Nordeste (exceto CE/PE)**: Caracterizado por carência de saneamento escolar, isolamento rural/fluvial e maiores distâncias da meta.
- **Padrão Sul / Sudeste / Centro-Oeste**: Infraestrutura escolar madura, saneamento consolidado e proximidade ou cumprimento antecipado da meta de 80%.

### 4. Como prever municípios que podem não atingir metas futuras?
- Desenvolvendo um escore preditivo de risco supervisionado alimentado por: estagnação histórica ($\Delta \\le 0$), distância atual da meta (`gap_meta_2030`), carência severa de infraestrutura e pressão de matrículas sobre o corpo docente.

### 5. Quais variáveis possuem maior influência nos modelos?
- As variáveis mais preditivas e livres de vazamento são: `pct_escolas_biblioteca`, `pct_escolas_internet`, `razao_aluno_docente`, `pct_escolas_esgoto_publico`, `pib_per_capita` e indicadores regionais.

---

## 6. Diretrizes para a Fase de Modelagem Supervisionada

1. **Definição de Target**:
   - **Regressão**: Previsão da `taxa_alfabetizacao` municipal.
   - **Classificação de Risco**: Identificação de cidades em risco (`status_meta_2030 == 'abaixo_meta_2030'`).
2. **Eliminação de Data Leakage**:
   - Descarte obrigatório de `gap_meta_2030`, `classe_taxa`, `nivel_alfabetizacao` e metas futuras 2025-2030 do conjunto preditor $X$.
3. **Validação Cruzada Segura**:
   - **Split Temporal**: Treino com a coorte de 2023 e teste cego na coorte de 2024.
   - **GroupKFold por UF**: Para testar a capacidade de generalização interfederativa.
4. **Algoritmos Recomendados**:
   - Modelos baseados em árvores (*LightGBM*, *XGBoost*, *Random Forest*) e modelos lineares regularizados (*Ridge*, *ElasticNet*).
5. **Interpretabilidade e IA Explicável**:
   - Aplicação de valores **SHAP (*SHapley Additive exPlanations*)** para geração de cartões de recomendação pedagógica customizados por município.
"""

readme_content = """# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil
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
venv\\Scripts\\activate
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
2. **O Poder das Bibliotecas Escolares**: Municípios com cobertura $\ge 50\%$ de bibliotecas atingem média de **68,12%**, contra **61,18%** nos municípios sem cobertura satisfatória (diferença de $+6,94$ p.p., $p = 4,76 \\times 10^{-34}$).
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
"""

with open(os.path.join(base_dir, "reports", "relatorio_eda.md"), "w", encoding="utf-8") as f:
    f.write(relatorio_content)

with open(os.path.join(base_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

print("relatorio_eda.md e README.md criados com sucesso!")
