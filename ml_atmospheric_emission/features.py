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


def create_state_columns(ds):
    ds = ds.copy()
    #Current state of the camera (only the past and the present: available at prediction time)
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds['score_atual'] = ds['y']
    ds['score_max_30'] = ds.set_index('Data')['y'].rolling('30min').max().values
    return ds


### Feature Enginnering Techinques ###########################################################################

### Feature Enginnering Techinques  1.0 : NaN Rows -----------------------------------------------------------
def InputingNanRows(ds):
    ds = ds.copy()
    


### Feature Enginnering Techinques  2.0 : Lags ---------------------------------------------------------------


### Feature Enginnering Techinques  3.0 : Sin(x),Cos(x) ------------------------------------------------------


### Feature Enginnering Techinques  4.0 : Derivates -----------------------------------------------------------

### Feature Enginnering Techinques  5.0: Solar Coordinates ----------------------------------------------------

### Feature Enginnering Techinques  6.0 : Climate Features ----------------------------------------------------


### Feature Enginnering Techinques  7.0 : Derivates -----------------------------------------------------------


### Feature Enginnering Techinques  8.0 : Models Features -----------------------------------------------------------


def main():
    ds = load_interim()
    ds = create_datetime_columns(ds)
    ds = create_target(ds)
    ds.to_csv(PATH_PROCESSED, index=False)


if __name__ == "__main__":
    main()
