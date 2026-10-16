# ml_atmospheric_emission

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Previsao da pluma visivel (score da camera, `nn_score_smog`) de 15 a 360 minutos a frente e recomendacao de
ajustes nas tags de processo para baixar o risco de pluma.

## Como usar

### 1. Instalar

```bash
git clone https://github.com/jmetrimiranda/ml_atmospheric_emission
cd ml_atmospheric_emission
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Treinar (com os dados de `data/raw/Tags_Pluma.xlsx`)

```bash
pluma treinar        # raw -> interim -> processed -> models/
```

### 3. Prever e recomendar (ultimo instante de `data/interim`)

```bash
pluma prever         # probabilidade de pluma em cada cabeca (15 a 360 min)
pluma recomendar     # ajustes nas tags que baixam o risco
pluma tudo           # os tres passos
```

### 4. Usar no Python (por exemplo, com dados novos de 5 em 5 minutos)

```python
from ml_atmospheric_emission.pipeline import prever, recomendar

#ds: colunas Data (hora local), nn_score_smog e as TAGS do config.py, com pelo menos os ultimos 30 minutos
previsoes = prever(ds)           #uma linha por instante, colunas prob_15 ... prob_360
recomendacao = recomendar(ds)    #dict: risco_atual, risco_recomendado, mudancas, risco_por_cabeca
```

### Onde estao as decisoes

Tudo o que foi decidido nos notebooks fica no `ml_atmospheric_emission/config.py`:

| Constante | O que e | Decidido em |
|---|---|---|
| `CABECAS_POSSIVEIS` | minutos a frente previstos | 3.0 |
| `PREVISOR_POR_CABECA` | persistencia (15 a 60 min) ou modelo (90 a 360 min) | 3.0, 4.0 e 5.0 |
| `MODELO_ATUAL` | colunas do modelo: tags + calendario + estado da camera | 4.0 |
| `ALGORITMO`, `FOLHA_MINIMA` | LightGBM com folhas de ~1 dia de dados | 5.0 |
| `VARIAVEIS_CONTROLE` | tags que a recomendacao pode mudar | 2.0 e 6.0 |

Com dados novos: coloque o Excel novo em `data/raw`, rode `pluma treinar`, e rode de novo os notebooks 3.0 a 6.0
para conferir se as decisoes continuam valendo.

Outra pasta de dados e modelos (pacote instalado fora do repositorio): `export PLUMA_PASTA=/caminho/da/pasta`
(a pasta precisa ter `data/` e `models/`).

### Limite importante da recomendacao

A previsao foi validada (notebooks 3.0 a 6.0). A recomendacao NAO: no notebook 6.0, os modelos de outro periodo nao
veem o risco cair com as mudancas sugeridas, e a direcao das tags muda de um periodo para outro
(`RECOMENDACAO_VALIDADA = False` no config). O modelo aprendeu associacoes, nao causa e efeito, e o sinal das tags
e fraco perto do score da camera. Use a recomendacao so como hipotese para testar com a operacao.

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

