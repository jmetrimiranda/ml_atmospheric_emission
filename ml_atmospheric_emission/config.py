from pathlib import Path

#Root folder of the project (works from notebook, terminal or Makefile)
PROJ_ROOT = Path(__file__).resolve().parents[1]

PATH_RAW = PROJ_ROOT / "data/raw/Tags_Pluma.xlsx"
PATH_INTERIM = PROJ_ROOT / "data/interim/dataset_interim.csv"
PATH_PROCESSED = PROJ_ROOT / "data/processed/dataset_processed.csv"
PATH_BASELINE = PROJ_ROOT / "reports/baseline_classificacao.csv"
PATH_MODELS = PROJ_ROOT / "models"
PATH_REPORTS = PROJ_ROOT / "reports"

#Process tags
TAGS = [
    '306-AIT-004-2', '306-AIT-004-5', '306FE002-AI001-R',
    '406-AI-004-2', '406-AI-004-4', '406-AIT-001'
]

#Calendar columns (context: they can not be changed by the operator)
#Mes is NOT here: with less than 2 months of data it works as a label of the period, not as climatology
CALENDARIO = ['Hora', 'Minuto', 'Dia_Semana']

#Columns used by the model
#nn_score_smog is NOT here: y is a copy of it (it would be the answer as input)
#int_sd and int_med are NOT here: they come from the camera image, not from the process
FEATURES = TAGS + CALENDARIO

#Current state of the camera (created in features.create_state_columns)
ESTADO = ['score_atual', 'score_max_30']

#Days with suspicious data: 22 hours of plume in a row, including the night
#(no other night has plume: the camera does not see the plume in the dark). CONFIRM with the team
DIAS_SUSPEITOS = ['2026-09-15', '2026-09-16']

#Tags the operator can act on (confirmed: all the process tags are controllable)
VARIAVEIS_CONTROLE = TAGS

#Models compared in notebooks/3.0-jm-escolha-horizontes.ipynb (name: columns used)
MODELOS = {
    'calendario': CALENDARIO,
    'base': TAGS + CALENDARIO,
    'base_estado': TAGS + CALENDARIO + ESTADO
}

#Forecast heads (minutes ahead of the plume prediction)
#Decision and why: reports/escolha_horizontes.md (made by notebooks/3.0-jm-escolha-horizontes.ipynb)
#Re-run the notebook when new variables or new data arrive, and update PREVISOR_POR_CABECA
CABECAS_POSSIVEIS = [15, 30, 60, 90, 120, 180, 240, 360]

#Forecaster of each head (indicator rule): the model is used only where it beats the persistence
#'persistencia' = score of the camera now | 'base_estado' = model of MODELOS
#Data from 2026-08-20 to 2026-10-13 (days of DIAS_SUSPEITOS removed)
PREVISOR_POR_CABECA = {
    15: 'persistencia',
    30: 'persistencia',
    60: 'persistencia',
    90: 'persistencia',
    120: 'persistencia',
    180: 'base_estado',
    240: 'base_estado',
    360: 'base_estado'
}
MODELO_ATUAL = TAGS + CALENDARIO + ESTADO