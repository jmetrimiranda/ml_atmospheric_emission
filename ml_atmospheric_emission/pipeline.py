import sys

from ml_atmospheric_emission import dataset, features
from ml_atmospheric_emission.config import MODELO_ATUAL, MODELOS, PREVISOR_POR_CABECA, VARIAVEIS_CONTROLE
from ml_atmospheric_emission.dataset import load_interim, load_processed
from ml_atmospheric_emission.features import create_model_features
from ml_atmospheric_emission.modeling import predict, recommend, train
from ml_atmospheric_emission.modeling.predict import load_head_models, predict_heads
from ml_atmospheric_emission.modeling.recommend import recommend_now

#Public use of the project (final version): 3 functions and the command 'pluma' of the terminal
#Input data (ds): a table every 5 minutes with the columns Data (local time), nn_score_smog and the TAGS
#of config.py (the same format of data/interim), with at least the last 30 minutes (score_max_30)


def treinar():
    #Trains everything with the decisions of config.py: raw -> interim -> processed -> models/
    dataset.main()
    features.main()
    train.main()


def prever(ds=None):
    #Probability of plume of each head (columns prob_15 ... prob_360) for each row of ds
    #ds=None: uses the interim of the project
    if ds is None:
        ds = load_interim()
    ds = create_model_features(ds)
    return predict_heads(ds, PREVISOR_POR_CABECA, MODELOS)


def recomendar(ds=None):
    #Changes in the controllable tags that lower the risk of plume at the LAST instant of ds
    #Returns a dict: risco_atual, risco_recomendado, mudancas, risco_por_cabeca
    #The limits of operation come from the history used in the training (data/processed)
    if ds is None:
        ds = load_interim()
    ds_agora = create_model_features(ds)
    modelos_cabeca = load_head_models(PREVISOR_POR_CABECA)
    return recommend_now(load_processed(), ds_agora, modelos_cabeca, MODELO_ATUAL, VARIAVEIS_CONTROLE)


def main():
    #Terminal: pluma treinar | pluma prever | pluma recomendar | pluma tudo
    comandos = ['treinar', 'prever', 'recomendar', 'tudo']
    if len(sys.argv) < 2 or sys.argv[1] not in comandos:
        print('Uso: pluma treinar | pluma prever | pluma recomendar | pluma tudo')
        return
    comando = sys.argv[1]
    if comando in ['treinar', 'tudo']:
        treinar()
    if comando in ['prever', 'tudo']:
        predict.main()
    if comando in ['recomendar', 'tudo']:
        recommend.main()


if __name__ == "__main__":
    main()
