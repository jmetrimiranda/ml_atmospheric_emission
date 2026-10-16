import json

import joblib

from ml_atmospheric_emission.config import MODELOS, PATH_MODELS, PREVISOR_POR_CABECA
from ml_atmospheric_emission.dataset import load_interim
from ml_atmospheric_emission.features import create_model_features, head_features


def load_head_models(previsores, pasta=PATH_MODELS):
    #Saved models of the heads that use a model (made by train.py): {minutos: model}
    modelos_cabeca = {}
    for minutos in previsores:
        if previsores[minutos] != 'persistencia':
            modelos_cabeca[minutos] = joblib.load(pasta / f'modelo_y_{minutos}.pkl')
    return modelos_cabeca


def load_thresholds(pasta=PATH_MODELS):
    #Threshold of the alarm of each head (made by train.py): {minutos: limiar}. None if there is no file
    caminho = pasta / 'limiares.json'
    if not caminho.exists():
        return None
    with open(caminho) as arquivo:
        dados = json.load(arquivo)
    limiares = {}
    for chave in dados:
        limiares[int(chave)] = dados[chave]
    return limiares


def predict_heads(ds, previsores, modelos, pasta=PATH_MODELS):
    #Plume probability for each head (columns prob_<minutos>), using the forecaster of each head:
    #'persistencia' = camera score now (column 'y'), otherwise the saved model of that head
    #and the alarm (columns alarme_<minutos>: 1 = probability >= threshold of the head, models/limiares.json)
    modelos_cabeca = load_head_models(previsores, pasta)
    limiares = load_thresholds(pasta)
    resultado = ds[['Data']].copy()
    for minutos in previsores:
        nome = previsores[minutos]
        if nome == 'persistencia':
            resultado['prob_' + str(minutos)] = ds['y'].values
        else:
            resultado['prob_' + str(minutos)] = modelos_cabeca[minutos].predict_proba(ds[head_features(modelos[nome], minutos)])[:, 1]
        if limiares is not None:
            resultado['alarme_' + str(minutos)] = (resultado['prob_' + str(minutos)] >= limiares[minutos]).astype(int)
    return resultado


def main():
    #Step 4 of the pipeline: probability of plume of each head at the LAST instant of the interim
    #(in production: put the newest data in the interim and run this file)
    ds = load_interim()
    ds = create_model_features(ds)
    resultado = predict_heads(ds, PREVISOR_POR_CABECA, MODELOS)
    ultima = resultado.iloc[-1]
    print('Previsao feita em', ultima['Data'], '(score da camera agora:', round(ds['y'].iloc[-1], 3), ')')
    limiares = load_thresholds()
    for minutos in PREVISOR_POR_CABECA:
        texto = '    ' + str(minutos) + ' min (' + PREVISOR_POR_CABECA[minutos] + '): probabilidade de pluma = ' + str(round(ultima['prob_' + str(minutos)], 3))
        if limiares is not None:
            if ultima['alarme_' + str(minutos)] == 1:
                texto = texto + ' | limiar ' + str(round(limiares[minutos], 3)) + ' | ALARME DE PLUMA'
            else:
                texto = texto + ' | limiar ' + str(round(limiares[minutos], 3)) + ' | sem alarme'
        print(texto)
    return resultado


if __name__ == "__main__":
    main()
