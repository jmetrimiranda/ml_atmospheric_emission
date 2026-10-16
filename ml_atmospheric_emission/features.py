from ml_atmospheric_emission.config import CABECAS_POSSIVEIS, DIAS_SUSPEITOS, PATH_PROCESSED
from ml_atmospheric_emission.dataset import load_interim, remove_days
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
#All the techniques only use the past and the present (available at prediction time)

### Feature Enginnering Techinques  1.0 : NaN Rows -----------------------------------------------------------
def InputingNanRows(ds, colunas, limite=6):
    ds = ds.copy()
    #Fills the NaN with the last known value (forward fill)
    #limite: at most 6 rows (30 minutes) after the last value, to not carry old values too far
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds[colunas] = ds[colunas].ffill(limit=limite)
    return ds


### Feature Enginnering Techinques  2.0 : Lags ---------------------------------------------------------------
def create_lag_columns(ds, colunas, lista_minutos):
    ds = ds.copy()
    #Value of each column some minutes ago (by the time, so a missing row gives NaN)
    ds = ds.sort_values('Data').reset_index(drop=True)
    for coluna in colunas:
        serie = ds.set_index('Data')[coluna]
        for minutos in lista_minutos:
            ds[coluna + '_lag_' + str(minutos)] = ds['Data'].map(serie.shift(minutos, freq='min'))
    return ds


### Feature Enginnering Techinques  3.0 : Sin(x),Cos(x) ------------------------------------------------------
def cyclic_feature(valores, periodo):
    #Cyclical feature of a cyclic variable: the value becomes a point on a circle (sin, cos)
    #sin(2*pi*x/periodo) and cos(2*pi*x/periodo): 23:55 is close to 00:00 and 359 degrees is close to 1 degree
    #Always the two together: sin alone gives the same value for two different moments of the cycle
    angulo = 2*np.pi*valores/periodo
    return np.sin(angulo), np.cos(angulo)


def create_cyclic_columns(ds):
    ds = ds.copy()
    #Hour of the day (period 24 h, with the minutes) and day of the week (period 7 days)
    hora_decimal = ds['Hora'] + ds['Minuto'] / 60
    ds['hora_sin'], ds['hora_cos'] = cyclic_feature(hora_decimal, 24)
    ds['dia_semana_sin'], ds['dia_semana_cos'] = cyclic_feature(ds['Dia_Semana'], 7)
    return ds


### Feature Enginnering Techinques  4.0 : Derivates -----------------------------------------------------------
def create_derivative_columns(ds, colunas, lista_minutos):
    ds = ds.copy()
    #Derivative: how much each column changed per minute in the last minutes, (x(t) - x(t - m)) / m
    ds = ds.sort_values('Data').reset_index(drop=True)
    for coluna in colunas:
        serie = ds.set_index('Data')[coluna]
        for minutos in lista_minutos:
            passado = ds['Data'].map(serie.shift(minutos, freq='min'))
            ds[coluna + '_der_' + str(minutos)] = (ds[coluna] - passado) / minutos
    return ds

### Feature Enginnering Techinques  5.0: Solar Coordinates ----------------------------------------------------
def solar_position(datas, latitude, longitude, fuso_horas):
    #Position of the sun at the times in datas (approximate NOAA equations, error below 0.5 degree)
    #latitude/longitude of the plant (south and west negative), fuso_horas = time zone of the times (Brasilia = -3)
    #Returns the elevation (degrees above the horizon) and the azimuth (degrees, 0 = north, 90 = east)
    dia_ano = datas.dt.dayofyear
    hora = datas.dt.hour + datas.dt.minute / 60
    gama = 2*np.pi/365 * (dia_ano - 1 + (hora - 12)/24)
    equacao_tempo = 229.18*(0.000075 + 0.001868*np.cos(gama) - 0.032077*np.sin(gama)
                            - 0.014615*np.cos(2*gama) - 0.040849*np.sin(2*gama))
    declinacao = (0.006918 - 0.399912*np.cos(gama) + 0.070257*np.sin(gama) - 0.006758*np.cos(2*gama)
                  + 0.000907*np.sin(2*gama) - 0.002697*np.cos(3*gama) + 0.00148*np.sin(3*gama))
    tempo_solar = hora*60 + equacao_tempo + 4*longitude - 60*fuso_horas
    #Solar time inside the day (0 to 1440 minutes): close to midnight it passes 1440 and the azimuth would flip
    tempo_solar = tempo_solar % 1440
    angulo_horario = np.radians(tempo_solar/4 - 180)
    lat = np.radians(latitude)
    cos_zenite = np.sin(lat)*np.sin(declinacao) + np.cos(lat)*np.cos(declinacao)*np.cos(angulo_horario)
    zenite = np.arccos(np.clip(cos_zenite, -1, 1))
    elevacao = 90 - np.degrees(zenite)
    cos_azimute = (np.sin(lat)*np.cos(zenite) - np.sin(declinacao)) / (np.cos(lat)*np.sin(zenite))
    azimute = np.degrees(np.arccos(np.clip(cos_azimute, -1, 1)))
    azimute = np.where(angulo_horario > 0, (azimute + 180) % 360, (540 - azimute) % 360)
    return np.asarray(elevacao), np.asarray(azimute)


def create_solar_columns(ds, latitude, longitude, fuso_horas):
    ds = ds.copy()
    #Solar coordinates NOW: elevation (not cyclic, it goes up and down between -90 and 90)
    #and azimuth (cyclic, period 360 degrees: sin and cos). The noon sun is to the north here
    elevacao, azimute = solar_position(ds['Data'], latitude, longitude, fuso_horas)
    ds['elevacao_solar'] = elevacao
    ds['azimute_sin'], ds['azimute_cos'] = cyclic_feature(azimute, 360)
    return ds
#Solar coordinates with lags and derivatives: create_lag_columns (2.0) and create_derivative_columns (4.0)
#on the columns of SOLAR (config.py). On the sin and cos, never on the degrees (359 -> 1 is not a jump of -358)


def create_solar_future_columns(ds, latitude, longitude, fuso_horas, cabecas):
    ds = ds.copy()
    #Solar coordinates at the instant being predicted (t + k): the sun position is known in advance, it is not a leak
    #One set of columns per head: elevacao_solar_futuro_60 = elevation of the sun at t + 60 min
    for minutos in cabecas:
        futuro = ds['Data'] + pd.Timedelta(minutes=minutos)
        elevacao, azimute = solar_position(futuro, latitude, longitude, fuso_horas)
        ds['elevacao_solar_futuro_' + str(minutos)] = elevacao
        ds['azimute_sin_futuro_' + str(minutos)], ds['azimute_cos_futuro_' + str(minutos)] = cyclic_feature(azimute, 360)
    return ds


def head_features(features, minutos):
    #Columns of the instant being predicted depend on the head: 'x_futuro' becomes 'x_futuro_60' in the 60 min head
    lista = []
    for coluna in features:
        if coluna.endswith('_futuro'):
            lista.append(coluna + '_' + str(minutos))
        else:
            lista.append(coluna)
    return lista


### Feature Enginnering Techinques  6.0 : Climate Features ----------------------------------------------------
#The climate comes from Open-Meteo (dataset.load_climate): hourly forecast made 1 day before each hour
def prepare_climate(ds_clima, fuso_horas):
    ds_clima = ds_clima.copy()
    #Time of the plant: UTC + fuso_horas (Brasilia: UTC - 3)
    ds_clima['Data'] = ds_clima['Data_UTC'] + pd.Timedelta(hours=fuso_horas)
    #Wind direction is cyclic (degrees): sin and cos BEFORE any lag, derivative or interpolation
    #(between 350 and 10 degrees the mean of the degrees is 180, the opposite direction)
    ds_clima['wind_direction_10m_sin'], ds_clima['wind_direction_10m_cos'] = cyclic_feature(ds_clima['wind_direction_10m'], 360)
    ds_clima = ds_clima.drop(columns=['Data_UTC', 'wind_direction_10m'])
    ds_clima = ds_clima.set_index('Data').sort_index()
    #From hourly to every 5 minutes: straight line between two hours of the forecast
    #(the whole forecast was published 1 day before, so using the next hour is not a leak)
    #limit=11: fills only between two hours that exist, not across hours missing in the API
    grade = pd.date_range(ds_clima.index.min(), ds_clima.index.max(), freq='5min')
    ds_clima = ds_clima.reindex(grade).interpolate(method='time', limit=11, limit_area='inside')
    return ds_clima


def create_climate_columns(ds, ds_clima, colunas):
    ds = ds.copy()
    #Climate NOW (t): forecast for the instant t, made 1 day before
    for coluna in colunas:
        ds[coluna] = ds['Data'].map(ds_clima[coluna]).values
    return ds
#Climate with lags and derivatives: create_lag_columns (2.0) and create_derivative_columns (4.0) on CLIMA,
#in hours (CLIMA_LAG_MINUTOS), because the climate comes once per hour


def create_climate_future_columns(ds, ds_clima, colunas, cabecas):
    ds = ds.copy()
    #Climate FORECAST for the instant being predicted (t + k), made 1 day before (published before t)
    #One set of columns per head: temperature_2m_futuro_60 = forecast of the temperature for t + 60 min
    for minutos in cabecas:
        futuro = ds['Data'] + pd.Timedelta(minutes=minutos)
        for coluna in colunas:
            ds[coluna + '_futuro_' + str(minutos)] = futuro.map(ds_clima[coluna]).values
    return ds


### Feature Enginnering Techinques  7.0 : Derivates -----------------------------------------------------------
#Same as 4.0 (the derivatives are in create_derivative_columns)


### Feature Enginnering Techinques  8.0 : Models Features -----------------------------------------------------------
#Persistence as a feature: create_state_columns (score_atual, score_max_30), tested with and without in the notebook 4.0
#Predictions of the heads before k as features (chain of heads): modeling/train.py, train_heads_chain_cv


### Steps of the chosen model (PREVISOR_POR_CABECA in config.py) ------------------------------------------
#Persistence heads (15 to 120 min) use only y (camera score now)
#Model heads (180 to 360 min, base_estado) use TAGS + CALENDARIO + ESTADO
#The other techniques (lags, sin/cos, solar, climate...) were tested in the notebook 4.0 and did not improve
def create_head_targets(ds, cabecas):
    ds = ds.copy()
    #Targets of the heads only (y_15, y_30, ...): camera score k minutes later, found by the time
    #(a missing row gives NaN, it never takes the next row by mistake)
    ds = ds.sort_values('Data').reset_index(drop=True)
    y_tempo = ds.set_index('Data')['y']
    for minutos in cabecas:
        ds['y_' + str(minutos)] = ds['Data'].map(y_tempo.shift(-minutos, freq='min'))
    return ds


def create_model_features(ds):
    #Columns needed to PREDICT at the instant t: calendar, y (persistence) and camera state
    ds = create_datetime_columns(ds)
    ds = create_target(ds)
    ds = create_state_columns(ds)
    return ds


def create_training_dataset(ds, cabecas):
    #Table to TRAIN and EVALUATE the chosen model (same order of the notebooks 3.0, 4.0 and 5.0):
    #calendar -> y -> without the suspicious days -> camera state -> targets of the heads
    ds = create_datetime_columns(ds)
    ds = create_target(ds)
    ds = remove_days(ds, DIAS_SUSPEITOS)
    ds = create_state_columns(ds)
    ds = create_head_targets(ds, cabecas)
    return ds


def main():
    #Step 2 of the pipeline: interim -> processed (features + targets of the heads)
    ds = load_interim()
    ds = create_training_dataset(ds, CABECAS_POSSIVEIS)
    ds.to_csv(PATH_PROCESSED, index=False)
    print('Processed salvo em', PATH_PROCESSED, '| linhas:', len(ds), '| colunas:', ds.shape[1])


if __name__ == "__main__":
    main()
