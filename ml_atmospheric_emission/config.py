from pathlib import Path

#Root folder of the project (works from notebook, terminal or Makefile)
PROJ_ROOT = Path(__file__).resolve().parents[1]

PATH_RAW = PROJ_ROOT / "data/raw/Tags_Pluma.xlsx"
PATH_INTERIM = PROJ_ROOT / "data/interim/dataset_interim.csv"
PATH_PROCESSED = PROJ_ROOT / "data/processed/dataset_processed.csv"
PATH_BASELINE = PROJ_ROOT / "reports/baseline_classificacao.csv"

#Columns used by the model
FEATURES = [
    '306-AIT-004-2', '306-AIT-004-5', '306FE002-AI001-R',
    '406-AI-004-2', '406-AI-004-4', '406-AIT-001',
    'Mes', 'Hora', 'Minuto', 'Dia_Semana', 'nn_score_smog'
]

#Process tags
TAGS = [
    '306-AIT-004-2', '306-AIT-004-5', '306FE002-AI001-R',
    '406-AI-004-2', '406-AI-004-4', '406-AIT-001'
]

#Calendar columns (context: they can not be changed by the operator)
#Mes is NOT here: with less than 2 months of data it works as a label of the period, not as climatology
CALENDARIO = ['Hora', 'Minuto', 'Dia_Semana']

FEATURES = TAGS + CALENDARIO

#Current state of the camera (created in features.create_state_columns)
ESTADO = ['score_atual', 'score_max_30']

#Days with suspicious data: 22 hours of plume in a row, including the night. CONFIRM with the team
DIAS_SUSPEITOS = ['2026-09-15', '2026-09-16']

#Tags the operator can act on (confirmed: all the process tags are controllable)
VARIAVEIS_CONTROLE = TAGS

#Models compared in notebooks/3.0-jm-escolha-horizontes.ipynb (name: columns used)
MODELOS = {
    'calendario': CALENDARIO,
    'base': TAGS + CALENDARIO,
    'base_estado': TAGS + CALENDARIO + ESTADO
}

#Forecast heads (minutes ahead). Decision and why: reports/escolha_horizontes.md
CABECAS_POSSIVEIS = [15, 30, 60, 90, 120, 180, 240, 360]

#Forecaster of each head (indicator rule): the model is used only where it beats the persistence
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