import pandas as pd

from ml_atmospheric_emission.config import PATH_EXTERNAL, PATH_INTERIM, PATH_PROCESSED, PATH_RAW


def load_raw(path=PATH_RAW):
    ds = pd.read_excel(path)
    return ds


def rename_columns(ds):
    #Conveting data-type for each columns and renaming
    ds = ds.rename(columns={
        'Unnamed: 0':'Data'
    })
    ds['Data'] = pd.to_datetime(ds['Data'])
    return ds


def filter_rows(ds):
    ds = ds[ds['nn_score_smog'].notna()]
    ds.drop(columns=['nn_ok_smog'],inplace=True)
    #raise NotImplementedError("Cole aqui o filtro, o y e o drop de nn_ok_smog da celula [55]")
    return ds


def load_interim(path=PATH_INTERIM):
    ds = pd.read_csv(path)
    ds['Data'] = pd.to_datetime(ds['Data'])
    return ds


def load_processed(path=PATH_PROCESSED):
    #Table made by features.main: features + targets of the heads of the chosen model
    ds = pd.read_csv(path)
    ds['Data'] = pd.to_datetime(ds['Data'])
    return ds


def round_float_columns(ds):
    ds = ds.copy()
    #Rounding float columns to 3 decimal places
    colunas = ds.columns
    types_ds = list(ds.dtypes)
    types_ds = [str(types_ds) for types_ds in types_ds]
    cont = 0
    for type_ds in types_ds:
        if type_ds == 'float64':
            ds[colunas[cont]] = ds[colunas[cont]].round(3)
        cont = cont+1
    return ds


def remove_days(ds, dias):
    #Removes whole days (for example, days with suspicious data)
    ds = ds.copy()
    dia = ds['Data'].dt.date.astype(str)
    ds = ds[~dia.isin(dias)].reset_index(drop=True)
    return ds


def download_climate(latitude, longitude, data_inicio, data_fim, variaveis):
    #Hourly climate of Open-Meteo at the plant position, times in UTC (column Data_UTC)
    #Previous Runs API, suffix _previous_day1: the forecast of each hour made 1 day BEFORE it.
    #Everything was already published before t, so it can be the climate now (t), in the past (lags)
    #and at the instant being predicted (t + k, k up to 6 h), with no information from after t.
    #Not the Historical Forecast API (the run of the hour uses observations of up to ~3 h later)
    #and not the Archive API (ERA5 reanalysis uses observations made after the hour and arrives ~5 days late)
    #The free API is for non commercial use: for the plant, check the license (API key) of Open-Meteo
    import requests
    url = 'https://previous-runs-api.open-meteo.com/v1/forecast'
    lista = [variavel + '_previous_day1' for variavel in variaveis]
    parametros = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': data_inicio,
        'end_date': data_fim,
        'hourly': ','.join(lista),
        'timezone': 'GMT'
    }
    resposta = requests.get(url, params=parametros, timeout=120)
    if resposta.status_code != 200:
        raise ValueError('Open-Meteo respondeu ' + str(resposta.status_code) + ': ' + resposta.text[:300])
    ds = pd.DataFrame(resposta.json()['hourly'])
    #Back to the names of the variables (without _previous_day1)
    ds.columns = [coluna.replace('_previous_day1', '') for coluna in ds.columns]
    ds = ds.rename(columns={'time': 'Data_UTC'})
    ds['Data_UTC'] = pd.to_datetime(ds['Data_UTC'])
    #Numbers as float (a variable that comes all empty would be read as text)
    for variavel in variaveis:
        ds[variavel] = ds[variavel].astype(float)
    return ds


def load_climate(latitude, longitude, data_inicio, data_fim, variaveis):
    #Downloads only once: the table is saved in data/external (one file per position) and read from there
    #If the saved table does not cover the period (new data arrived) or the variables, it is downloaded again
    caminho = PATH_EXTERNAL / ('clima_open_meteo_' + str(latitude) + '_' + str(longitude) + '.csv')
    if caminho.exists():
        ds = pd.read_csv(caminho)
        ds['Data_UTC'] = pd.to_datetime(ds['Data_UTC'])
        cobre = ds['Data_UTC'].min() <= pd.Timestamp(data_inicio) and ds['Data_UTC'].max() >= pd.Timestamp(data_fim)
        if cobre and set(variaveis).issubset(ds.columns):
            print('Clima lido de', caminho, '| linhas:', len(ds))
            return ds
    ds = download_climate(latitude, longitude, data_inicio, data_fim, variaveis)
    PATH_EXTERNAL.mkdir(parents=True, exist_ok=True)
    ds.to_csv(caminho, index=False)
    print('Clima baixado do Open-Meteo e salvo em', caminho, '| linhas:', len(ds))
    return ds


def main():
    #Step 1 of the pipeline: raw (excel) -> interim (csv). Next: features.py, train.py, predict.py
    ds = load_raw()
    ds = rename_columns(ds)
    ds  = round_float_columns(ds)
    ds = filter_rows(ds)
    PATH_INTERIM.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(PATH_INTERIM, index=False)
    print('Interim salvo em', PATH_INTERIM, '| linhas:', len(ds))


if __name__ == "__main__":
    main()
