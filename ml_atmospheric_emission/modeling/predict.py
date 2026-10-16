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


def predict_heads(ds, previsores, modelos, pasta=PATH_MODELS):
    #Plume probability for each head (one column per head), using the forecaster of each head:
    #'persistencia' = camera score now (column 'y'), otherwise the saved model of that head
    modelos_cabeca = load_head_models(previsores, pasta)
    resultado = ds[['Data']].copy()
    for minutos in previsores:
        nome = previsores[minutos]
        if nome == 'persistencia':
            resultado['prob_' + str(minutos)] = ds['y'].values
        else:
            resultado['prob_' + str(minutos)] = modelos_cabeca[minutos].predict_proba(ds[head_features(modelos[nome], minutos)])[:, 1]
    return resultado


def main():
    #Step 4 of the pipeline: probability of plume of each head at the LAST instant of the interim
    #(in production: put the newest data in the interim and run this file)
    ds = load_interim()
    ds = create_model_features(ds)
    resultado = predict_heads(ds, PREVISOR_POR_CABECA, MODELOS)
    ultima = resultado.iloc[-1]
    print('Previsao feita em', ultima['Data'], '(score da camera agora:', round(ds['y'].iloc[-1], 3), ')')
    for minutos in PREVISOR_POR_CABECA:
        print('   ', minutos, 'min (' + PREVISOR_POR_CABECA[minutos] + '): probabilidade de pluma =', round(ultima['prob_' + str(minutos)], 3))
    return resultado


if __name__ == "__main__":
    main()
