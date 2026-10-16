import joblib

from ml_atmospheric_emission.config import MODELOS, PATH_MODELS, PREVISOR_POR_CABECA
from ml_atmospheric_emission.dataset import load_interim
from ml_atmospheric_emission.features import create_model_features, head_features


def predict_heads(ds, previsores, modelos):
    #Plume score for each head (one column per head), using the forecaster of each head:
    #'persistencia' = camera score now (column 'y'), otherwise the saved model of that head
    resultado = ds[['Data']].copy()
    for minutos in previsores:
        nome = previsores[minutos]
        if nome == 'persistencia':
            resultado['prob_' + str(minutos)] = ds['y'].values
        else:
            model = joblib.load(PATH_MODELS / f'modelo_y_{minutos}.pkl')
            resultado['prob_' + str(minutos)] = model.predict_proba(ds[head_features(modelos[nome], minutos)])[:, 1]
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
