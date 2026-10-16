import joblib

from ml_atmospheric_emission.config import PATH_MODELS


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
            resultado['prob_' + str(minutos)] = model.predict_proba(ds[modelos[nome]])[:, 1]
    return resultado
