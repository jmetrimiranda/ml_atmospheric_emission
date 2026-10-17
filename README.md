# ml_atmospheric_emission

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Previsao da pluma visivel (score da camera, `nn_score_smog`) de 15 a 360 minutos a frente e recomendacao de
ajustes nas tags de processo para baixar o risco de pluma.

## Como usar

### 1. Instalar (uma vez)

```bash
git clone https://github.com/jmetrimiranda/ml_atmospheric_emission
cd ml_atmospheric_emission
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # instala as dependencias e cria o comando pluma
```

### 2. Rodar (terminal)

```bash
pluma treinar      # Excel de data/raw -> data/interim -> data/processed -> models/ (modelos + limiares)
pluma prever       # probabilidade e alarme de pluma em cada cabeca, no ultimo instante de data/interim
pluma recomendar   # ajustes nas tags que baixam o risco (hipotese, ver o limite abaixo)
pluma tudo         # os tres
```

`make pipeline` faz o mesmo que `pluma tudo`.

### 3. Usar no Python

```python
from ml_atmospheric_emission.pipeline import treinar, prever, recomendar

#ds: tabela de 5 em 5 minutos, crua (as colunas de data/raw) ou no formato de data/interim,
#com pelo menos os ultimos 30 minutos
treinar()                      #ou treinar(ds_historico) para treinar com uma tabela em vez do Excel
previsoes = prever(ds)         #colunas prob_15 ... prob_360 e alarme_15 ... alarme_360 (1 = alarme)
recomendacao = recomendar(ds)  #dict: risco_atual, risco_recomendado, mudancas, risco_por_cabeca

#Tabela crua (as mesmas colunas de data/raw) -> as duas tabelas do Databricks
from ml_atmospheric_emission.pipeline import tabela_previsao, tabela_recomendacao
tabela_1 = tabela_previsao(ds_bruto)        #uma linha por instante
tabela_2 = tabela_recomendacao(ds_bruto)    #uma linha por tag a mudar, nos instantes com alarme
```

### 4. O que sai

`prob_<k>` = probabilidade de pluma k minutos a frente (persistencia: o score da camera agora).
`alarme_<k>` = 1 se `prob_<k>` >= limiar da cabeca. Os limiares sao escolhidos no treino com o maximo F1
(`BETA_LIMIAR = 1`) e ficam em `models/limiares.json`.

| Cabeca | Previsor | Teste (hold out): AP | precisao | recall | F1 |
|---|---|---|---|---|---|
| 15 min | persistencia | 0,72 | 0,71 | 0,71 | 0,71 |
| 30 min | persistencia | 0,63 | 0,64 | 0,64 | 0,64 |
| 60 min | persistencia | 0,47 | 0,53 | 0,55 | 0,54 |
| 90 min | LightGBM | 0,51 | 0,53 | 0,59 | 0,56 |
| 120 min | LightGBM | 0,47 | 0,38 | 0,75 | 0,51 |
| 180 min | LightGBM | 0,45 | 0,30 | 0,92 | 0,45 |
| 240 min | LightGBM | 0,41 | 0,29 | 0,85 | 0,43 |
| 360 min | LightGBM | 0,51 | 0,31 | 0,93 | 0,46 |
Detalhes: `notebooks/7.0-jm-performance.ipynb`.

### 5. Databricks: previsao e recomendacao a cada 5 minutos (sem treinar)

O notebook `databricks/pluma_job_5min.py` roda como Job a cada 5 minutos: le as linhas novas da tabela crua
(as mesmas colunas de `data/raw`), usa os modelos ja treinados e grava duas tabelas Delta sem duplicar (MERGE):

| Tabela | Uma linha por | Colunas |
|---|---|---|
| previsao | instante de 5 min | `Data`, `score_camera`, `prob_<k>` e `alarme_<k>` (k = 15 ... 360), `versao_modelo` |
| recomendacao | tag a mudar, nos instantes com alarme | `Data`, `risco_atual`, `risco_recomendado`, `variavel`, `valor_atual`, `valor_recomendado`, `direcao`, `validada`, `versao_modelo` |

**Jeito A (mais simples): Git folder.** No Databricks: Workspace > Create > Git folder > URL deste repositorio.
O codigo, os `models/` e o `data/processed` ja estao no repositorio. No notebook, troque a linha do `%pip` por
`%pip install /Workspace/Users/<voce>/ml_atmospheric_emission` e use essa mesma pasta no widget `pasta`.

**Jeito B (producao): Volume + wheel.** `make wheel` e suba para um Volume do Unity Catalog:

```
/Volumes/<catalogo>/<schema>/pluma/ml_atmospheric_emission-1.1.0-py3-none-any.whl
/Volumes/<catalogo>/<schema>/pluma/models/                       (modelo_y_*.pkl e limiares.json)
/Volumes/<catalogo>/<schema>/pluma/data/processed/dataset_processed.csv   (historico: limites da recomendacao)
```

Depois, nos dois jeitos:
1. Abra `databricks/pluma_job_5min.py`, preencha os widgets (pasta, tabela crua, coluna de tempo, tabelas de saida) e rode uma vez.
2. Se der erro ao carregar os `.pkl` (versoes diferentes de numpy, scikit-learn ou LightGBM no cluster), rode uma vez com
   `treinar = sim`: refaz os `.pkl` no proprio cluster em poucos segundos, a partir do `data/processed`.
3. Workflows > Create job > tarefa Notebook com esse notebook > agendamento a cada 5 minutos
   (cron Quartz `0 0/5 * * * ?`). Use computacao serverless: um cluster classico demora minutos para ligar.

A coluna de tempo da tabela crua deve estar no horario local da planta, como no Excel. A simulacao local do Job
(sem Spark) esta no `notebooks/8.0-jm-simulacao-databricks.ipynb`.

### Onde estao as decisoes

Tudo o que foi decidido nos notebooks fica no `ml_atmospheric_emission/config.py`:

| Constante | O que e | Decidido em |
|---|---|---|
| `CABECAS_POSSIVEIS` | minutos a frente previstos | 3.0 |
| `PREVISOR_POR_CABECA` | persistencia (15 a 60 min) ou modelo (90 a 360 min) | 3.0, 4.0 e 5.0 |
| `MODELO_ATUAL` | colunas do modelo: tags + calendario + estado da camera | 4.0 |
| `ALGORITMO`, `FOLHA_MINIMA` | LightGBM com folhas de ~1 dia de dados | 5.0 |
| `VARIAVEIS_CONTROLE`, `RECOMENDACAO_ATIVA` | tags que a recomendacao pode mudar, recomendacao ligada | 2.0 e 6.0 |
| `BETA_LIMIAR` | criterio do limiar do alarme (1 = F1) | 7.0 |

Com dados novos: coloque o Excel novo em `data/raw` (ou use `treinar(ds)`), rode `pluma treinar` e rode de novo os
notebooks 3.0 a 7.0 para conferir se as decisoes continuam valendo.

### Limite importante da recomendacao

A previsao foi validada (notebooks 3.0 a 7.0). A recomendacao esta ligada (`RECOMENDACAO_ATIVA = True`), mas NAO foi
validada: no notebook 6.0, os modelos de outro periodo nao veem o risco cair com as mudancas sugeridas, e a direcao das
tags muda de um periodo para outro. O modelo aprendeu associacoes, nao causa e efeito. Por isso cada recomendacao sai
com um aviso: use como hipotese para testar com a operacao (uma tag por vez), nao como ordem.

### Notebooks

| Notebook | O que faz |
|---|---|
| 0.0, 1.0 | ETL e exploracao |
| 2.0 | existe sinal das tags no futuro? primeira recomendacao |
| 3.0 | escolha das cabecas: modelo x persistencia |
| 4.0 | escolha das features (fatorial, solar, clima, persistencia, cadeia) |
| 5.0 | escolha do algoritmo (LightGBM, XGBoost, Gradient Boosting, Random Forest) |
| 6.0 | pipeline final e validacao da recomendacao |
| 7.0 | performance da versao final no teste (AP, precisao, recall, F1) e escolha do limiar |
| 8.0 | simulacao local do Job do Databricks (as duas tabelas, a cada 5 minutos) |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ml_atmospheric_emission and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ml_atmospheric_emission   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ml_atmospheric_emission a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

