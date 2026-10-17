import sys
from importlib.metadata import version

import pandas as pd

from ml_atmospheric_emission import dataset, features
from ml_atmospheric_emission.config import MODELO_ATUAL, MODELOS, PATH_INTERIM, PREVISOR_POR_CABECA, RECOMENDACAO_ATIVA, VARIAVEIS_CONTROLE
from ml_atmospheric_emission.dataset import load_interim, load_processed, prepare_raw
from ml_atmospheric_emission.features import create_model_features
from ml_atmospheric_emission.modeling import predict, recommend, train
from ml_atmospheric_emission.modeling.predict import load_head_models, predict_heads
from ml_atmospheric_emission.modeling.recommend import recommend_now, recommend_rows

#Public use of the project (final version): functions for Python/Databricks and the command 'pluma' of the terminal
#Input data (ds): a table every 5 minutes in the RAW format (the same columns of data/raw: time in the first column,
#nn_ok_smog, nn_score_smog, int_sd, int_med and the TAGS) or in the interim format (column Data, without nn_ok_smog).
#For each instant the model needs the last 30 minutes before it (score_max_30)


def treinar(ds=None):
    #Trains everything with the decisions of config.py: raw -> interim -> processed -> models/
    #ds=None: reads the Excel of data/raw | ds = raw table (for example a table of Databricks)
    if ds is None:
        dataset.main()
    else:
        ds = prepare_raw(ds)
        PATH_INTERIM.parent.mkdir(parents=True, exist_ok=True)
        ds.to_csv(PATH_INTERIM, index=False)
        print('Interim salvo em', PATH_INTERIM, '| linhas:', len(ds))
    features.main()
    train.main()


def prever(ds=None):
    #Probability of plume of each head (columns prob_15 ... prob_360) and alarm (alarme_15 ... alarme_360)
    #for each row of ds. ds=None: uses the interim of the project
    if ds is None:
        ds = load_interim()
    ds = create_model_features(prepare_raw(ds))
    return predict_heads(ds, PREVISOR_POR_CABECA, MODELOS)


def recomendar(ds=None):
    #Changes in the controllable tags that lower the risk of plume at the LAST instant of ds
    #Returns a dict: risco_atual, risco_recomendado, mudancas, risco_por_cabeca
    #The limits of operation come from the history used in the training (data/processed)
    if ds is None:
        ds = load_interim()
    ds_agora = create_model_features(prepare_raw(ds))
    modelos_cabeca = load_head_models(PREVISOR_POR_CABECA)
    return recommend_now(load_processed(), ds_agora, modelos_cabeca, MODELO_ATUAL, VARIAVEIS_CONTROLE)


def tabela_previsao(ds_bruto, a_partir_de=None):
    #TABLE 1 (Databricks): one row per instant of 5 minutes with the camera score now,
    #the probability and the alarm of each head and the version of the model
    #a_partir_de: only the instants AFTER this time (the older rows are only the 30 minutes of history)
    ds = create_model_features(prepare_raw(ds_bruto))
    previsoes = predict_heads(ds, PREVISOR_POR_CABECA, MODELOS)
    previsoes.insert(1, 'score_camera', ds['y'].values)
    if a_partir_de is not None:
        previsoes = previsoes[previsoes['Data'] > pd.Timestamp(a_partir_de)]
    previsoes['versao_modelo'] = version('ml_atmospheric_emission')
    return previsoes.reset_index(drop=True)


def tabela_recomendacao(ds_bruto, a_partir_de=None, so_alarme=True):
    #TABLE 2 (Databricks): one row per tag to change, for each instant (Data, risks, tag, current and
    #recommended value, direction, validada). so_alarme=True: only the instants with alarm in some head that uses
    #a model (where the risk matters). a_partir_de: only the instants AFTER this time
    ds = create_model_features(prepare_raw(ds_bruto))
    previsoes = predict_heads(ds, PREVISOR_POR_CABECA, MODELOS)
    selecionar = pd.Series(True, index=ds.index)
    if a_partir_de is not None:
        selecionar = selecionar & (ds['Data'] > pd.Timestamp(a_partir_de))
    if so_alarme:
        com_alarme = pd.Series(False, index=ds.index)
        for minutos in PREVISOR_POR_CABECA:
            if PREVISOR_POR_CABECA[minutos] != 'persistencia' and 'alarme_' + str(minutos) in previsoes.columns:
                com_alarme = com_alarme | (previsoes['alarme_' + str(minutos)] == 1)
        selecionar = selecionar & com_alarme
    modelos_cabeca = load_head_models(PREVISOR_POR_CABECA)
    tabela = recommend_rows(load_processed(), ds[selecionar], modelos_cabeca, MODELO_ATUAL, VARIAVEIS_CONTROLE)
    tabela['versao_modelo'] = version('ml_atmospheric_emission')
    return tabela


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
    if comando == 'recomendar' or (comando == 'tudo' and RECOMENDACAO_ATIVA):
        recommend.main()


if __name__ == "__main__":
    main()
