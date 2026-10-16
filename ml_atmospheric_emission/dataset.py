import pandas as pd

from ml_atmospheric_emission.config import PATH_RAW, PATH_INTERIM


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


def main():
    ds = load_raw()
    ds = rename_columns(ds)
    ds  = round_float_columns(ds)
    ds = filter_rows(ds)
    ds.to_csv(PATH_INTERIM, index=False)


if __name__ == "__main__":
    main()
