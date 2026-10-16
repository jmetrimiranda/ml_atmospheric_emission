import os
from pathlib import Path

#Root folder of the project (works from notebook, terminal or Makefile)
#Environment variable PLUMA_PASTA: another folder with data/ and models/ (when the package is installed elsewhere)
PROJ_ROOT = Path(os.environ.get('PLUMA_PASTA', Path(__file__).resolve().parents[1]))

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
#The recommendation did NOT pass the validation of the notebook 6.0: the models of another period do not see the
#risk going down, the opposite change helps the same, and the direction of the tags changes between the periods.
#Use it only as a hypothesis to test in the plant. Change to True only after a validation that passes
RECOMENDACAO_VALIDADA = False

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
#'persistencia' = score of the camera now | 'base_estado' = model of MODELOS (algorithm ALGORITMO)
#Data from 2026-08-20 to 2026-10-13 (days of DIAS_SUSPEITOS removed). Decision and why:
#  3.0: the model beats the persistence from 180 min (reports/escolha_horizontes.md)
#  4.0: no group of features improved the model (reports/escolha_features.csv)
#  5.0: with leaves of ~1 day (FOLHA_MINIMA) the LightGBM also beats the persistence in 90 and 120 min
#       (CV with Reality Check + Holm, confirmed in the hold out: reports/escolha_algoritmo.csv)
PREVISOR_POR_CABECA = {
    15: 'persistencia',
    30: 'persistencia',
    60: 'persistencia',
    90: 'base_estado',
    120: 'base_estado',
    180: 'base_estado',
    240: 'base_estado',
    360: 'base_estado'
}
#Columns of the model of the heads that use a model (the same of MODELOS['base_estado'])
MODELO_ATUAL = MODELOS['base_estado']

#Algorithms compared in notebooks/5.0-jm-escolha-algoritmo.ipynb (modeling/train.py, create_algorithm)
ALGORITMOS = ['lightgbm', 'xgboost', 'gradient_boosting', 'random_forest']
#Algorithm of the heads that use a model (base_estado). Decision and why: reports/escolha_algoritmo.csv
ALGORITMO = 'lightgbm'
#Against overfitting: each leaf of the trees has at least ~1 day of data (288 rows of 5 min),
#so a leaf can not memorize one plume episode (a few hours). With 100 rows the train AP was ~0.86 and the CV ~0.17
FOLHA_MINIMA = 300

#Feature engineering tested in notebooks/4.0-jm-escolha-melhores-features.ipynb
LAG_MINUTOS = [15, 30, 60]
DERIVADA_MINUTOS = [15, 60]
#Position of the plant to create the solar coordinates (south and west negative)
LATITUDE = -20.79
LONGITUDE = -40.58
#Time zone of the column 'Data' (Brasilia = -3, no daylight saving time since 2019)
FUSO_HORAS = -3
#Solar coordinates (made by features.create_solar_columns): the azimuth is cyclic, so it goes as sin and cos
SOLAR = ['elevacao_solar', 'azimute_sin', 'azimute_cos']

#Climate of Open-Meteo (data/external): hourly variables, names of the API
#wind_direction_10m is cyclic (degrees): the model receives its sin and cos, never the raw degrees
PATH_EXTERNAL = PROJ_ROOT / "data/external"
CLIMA_VARIAVEIS = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'pressure_msl', 'cloud_cover',
    'precipitation', 'wind_speed_10m', 'wind_direction_10m', 'shortwave_radiation'
]
#Climate columns used by the model (made by features.prepare_climate)
CLIMA = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'pressure_msl', 'cloud_cover',
    'precipitation', 'wind_speed_10m', 'wind_direction_10m_sin', 'wind_direction_10m_cos', 'shortwave_radiation'
]
#The climate is hourly: lags and derivatives in hours (15 minutes ago is the same hour, the same value)
CLIMA_LAG_MINUTOS = [60, 180]
CLIMA_DERIVADA_MINUTOS = [60, 180]
