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