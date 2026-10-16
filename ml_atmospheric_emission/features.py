from ml_atmospheric_emission.config import PATH_PROCESSED
from ml_atmospheric_emission.dataset import load_interim
from ml_atmospheric_emission.util import convert_to_ordinal 
import numpy as np
import pandas as pd

def create_datetime_columns(ds):
    ds = ds.copy()
    #Creating datetime columns
    ds['Mes'] = ds['Data'].dt.month
    ds['Hora'] = ds['Data'].dt.hour
    ds['Minuto'] = ds['Data'].dt.minute
    #ds['Segundos'] = ds['Data'].dt.seconds
    ds['Dia_Semana'] = ds['Data'].dt.dayofweek
    return ds
def create_target(ds):
    ds = ds.copy()
    #ds['y'] = ds.apply(convert_to_ordinal,axis = 1)
    #ds.rename(columns={'nn_score_smog':'y'},inplace=True)
    ds['y'] = ds['nn_score_smog']
    return ds 
    

def create_future_targets(ds,qtd_deslocamento):
    ds = ds.copy()
    #Sorting from oldest to newest
    ds = ds.sort_values('Data').reset_index(drop=True)
    #y indexed by time
    y_tempo = ds.set_index('Data')['y']
    vetor_5 = pd.Series(np.arange(1, qtd_deslocamento + 1, 1))
    for deslocamento_ in vetor_5:
        columns_name = f'y_{deslocamento_*5}'
        #print(columns_name)
        ds[columns_name] = ds['Data'].map(y_tempo.shift(-deslocamento_, freq='5min'))
    return ds


def main():
    ds = load_interim()
    ds = create_datetime_columns(ds)
    ds = create_target(ds)
    ds = create_future_targets(ds)
    ds.to_csv(PATH_PROCESSED, index=False)


if __name__ == "__main__":
    main()
