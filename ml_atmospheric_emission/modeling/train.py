import joblib
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor

import pandas as pd

from ml_atmospheric_emission.config import ALGORITMO, FEATURES, FOLHA_MINIMA, MODELOS, PATH_MODELS, PREVISOR_POR_CABECA
from ml_atmospheric_emission.dataset import load_processed
from ml_atmospheric_emission.features import head_features


def create_model(class_weight='balanced', sorteio=False):
    #Tree model with conservative parameters to avoid overfitting
    #class_weight=None gives probabilities closer to the real chance of smog (used to recommend)
    #sorteio=False (default): no random draw of rows and columns (subsample = colsample_bytree = 1), so the same data
    #always gives the same model and a comparison of features changes ONLY the features
    #(with the draw of 0.8, changing only the random_state gave "significant improvement" in 19% of the tests,
    #and without the draw the AP of the CV was higher in the 8 heads). sorteio=True: the old version
    if sorteio:
        fracao = 0.8
    else:
        fracao = 1.0
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=100,
        subsample=fracao,
        subsample_freq=1,
        colsample_bytree=fracao,
        reg_lambda=1.0,
        class_weight=class_weight,
        random_state=42,
        verbose=-1
    )
    return model


### Algorithms compared in the notebook 5.0 (all with parameters against overfitting) ---------------------
#The same brake for all: each leaf with at least FOLHA_MINIMA rows (~1 day), shallow trees (boosting: depth 3,
#200 trees, learning rate 0.05) and no random draw of rows and columns in the boosting (same data = same model)
def create_lightgbm():
    #LightGBM: boosting, the tree grows leaf by leaf (num_leaves=7 = the leaves of a tree of depth 3)
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=FOLHA_MINIMA,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    return model


def create_xgboost():
    #XGBoost: boosting, the tree grows level by level
    #min_child_weight is a sum of p*(1-p) (p = probability of plume, ~0.08): FOLHA_MINIMA / 12 is ~FOLHA_MINIMA rows
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=FOLHA_MINIMA / 12,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    return model


def create_gradient_boosting():
    #Gradient Boosting of sklearn (the classic one): it does not accept NaN, so the NaN of the tags
    #receive the median of the TRAIN (SimpleImputer inside the pipeline: fitted only with the train rows)
    model = make_pipeline(
        SimpleImputer(strategy='median'),
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=FOLHA_MINIMA,
            random_state=42
        )
    )
    return model


def create_random_forest():
    #Random Forest: 300 independent trees, the probability is the mean of the trees. Each tree draws rows and
    #columns (it is the method itself); 300 trees make the result stable. Leaves with at least FOLHA_MINIMA rows
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=FOLHA_MINIMA,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    return model


def create_algorithm(nome):
    #New model of the algorithm 'nome' (names of ALGORITMOS in config.py)
    #class_weight=None everywhere: the output is closer to the real probability of plume
    if nome == 'lightgbm':
        return create_lightgbm()
    if nome == 'xgboost':
        return create_xgboost()
    if nome == 'gradient_boosting':
        return create_gradient_boosting()
    if nome == 'random_forest':
        return create_random_forest()
    raise ValueError('Algoritmo desconhecido: ' + nome)


def train_algorithms_cv(ds, features, cabecas, algoritmos, n_splits=5, limiar=0.5):
    #Time series CV of each algorithm in each head, with the SAME rows and folds for all (paired comparison)
    #ds: train period with the columns y_15, y_30, ... | features: columns of the model | cabecas: minutes ahead
    #Returns:
    #  tabelas = {minutos: table with fold, Data, y_true, persistencia and one column per algorithm (probability)}
    #  ap_treino = table with the AP in the TRAIN rows (mean of the folds): far above the CV AP = overfitting
    tabelas = {}
    linhas_treino = []
    for minutos in cabecas:
        target = 'y_' + str(minutos)
        dx = ds[ds[target].notna()].sort_values('Data').reset_index(drop=True)
        y = (dx[target] >= limiar).astype(int)
        colunas = head_features(features, minutos)
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=minutos // 5)
        lista = []
        soma_treino = {}
        for nome in algoritmos:
            soma_treino[nome] = 0
        fold = 0
        for idx_train, idx_val in tscv.split(dx):
            if y.iloc[idx_train].nunique() < 2 or y.iloc[idx_val].sum() == 0:
                continue
            dy = pd.DataFrame({
                'fold': fold,
                'Data': dx['Data'].iloc[idx_val].values,
                'y_true': y.iloc[idx_val].values,
                'persistencia': dx['y'].iloc[idx_val].values
            })
            for nome in algoritmos:
                model = create_algorithm(nome)
                model.fit(dx[colunas].iloc[idx_train], y.iloc[idx_train])
                dy[nome] = model.predict_proba(dx[colunas].iloc[idx_val])[:, 1]
                prob_train = model.predict_proba(dx[colunas].iloc[idx_train])[:, 1]
                soma_treino[nome] = soma_treino[nome] + average_precision_score(y.iloc[idx_train], prob_train)
            lista.append(dy)
            fold = fold+1
        tabelas[minutos] = pd.concat(lista).reset_index(drop=True)
        linha = {'minutos': minutos}
        for nome in algoritmos:
            linha[nome] = soma_treino[nome] / fold
        linhas_treino.append(linha)
        print(minutos, 'min: ok')
    ap_treino = pd.DataFrame(linhas_treino)
    return tabelas, ap_treino


def create_regressor():
    #XGBoost with fixed conservative parameters to avoid overfitting
    #reg:logistic keeps the prediction between 0 and 1, like nn_score_smog
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='reg:logistic',
        random_state=42,
        n_jobs=-1
    )
    return model


def train_model_experiment(ds, target, features=FEATURES, test_size=0.2, n_splits=5, gap=0, limiar=0.5):
    ds = ds.copy()
    #Sorting by time and removing rows without target
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds[ds[target].notna()].reset_index(drop=True)
    X = ds[features]
    #Discrete target: same threshold of the neural network
    y = (ds[target] >= limiar).astype(int)

    #Hold out: the last part of the period is the test, with a gap between train and test
    corte = int(len(ds) * (1 - test_size))
    X_train = X.iloc[:corte - gap]
    y_train = y.iloc[:corte - gap]
    X_test = X.iloc[corte:]
    y_test = y.iloc[corte:]

    #Time series cross validation only inside the train period
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    f1_cv = []
    for idx_train, idx_val in tscv.split(X_train):
        #Skipping folds with only one class in train
        if y_train.iloc[idx_train].nunique() < 2:
            continue
        model = create_model()
        model.fit(X_train.iloc[idx_train], y_train.iloc[idx_train])
        y_pred_val = model.predict(X_train.iloc[idx_val])
        f1_cv.append(f1_score(y_train.iloc[idx_val], y_pred_val, zero_division=0))

    #Final model trained with all the train period
    model = create_model()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metricas = {
        'target': target,
        'f1_score_train': round(f1_score(y_train, y_pred_train, zero_division=0), 3),
        'f1_score_test': round(f1_score(y_test, y_pred_test, zero_division=0), 3),
        'recall_train': round(recall_score(y_train, y_pred_train, zero_division=0), 3),
        'recall_test': round(recall_score(y_test, y_pred_test, zero_division=0), 3),
        'precision_train': round(precision_score(y_train, y_pred_train, zero_division=0), 3),
        'precision_test': round(precision_score(y_test, y_pred_test, zero_division=0), 3),
        'f1_score_cv': round(np.mean(f1_cv), 3),
        'f1_score_cv_std': round(np.std(f1_cv), 3)
    }

    print('Modelo:', target)
    print('f1_score_train:', metricas['f1_score_train'])
    print('f1_score_test:', metricas['f1_score_test'])
    print('recall_train:', metricas['recall_train'])
    print('recall_test:', metricas['recall_test'])
    print('precision_train:', metricas['precision_train'])
    print('precision_test:', metricas['precision_test'])
    print('f1_score_cv:', metricas['f1_score_cv'], '+/-', metricas['f1_score_cv_std'])
    print()
    return model, metricas


def train_regressor_experiment(ds, target, features=FEATURES, test_size=0.2, n_splits=5, gap=0, limiar=0.5, discreto=False):
    ds = ds.copy()
    #Sorting by time and removing rows without target
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds[ds[target].notna()].reset_index(drop=True)
    X = ds[features]
    y = ds[target]

    #Hold out: the last part of the period is the test, with a gap between train and test
    corte = int(len(ds) * (1 - test_size))
    X_train = X.iloc[:corte - gap]
    y_train = y.iloc[:corte - gap]
    X_test = X.iloc[corte:]
    y_test = y.iloc[corte:]

    #Real classes using the same threshold of the neural network
    y_train_class = (y_train >= limiar).astype(int)
    y_test_class = (y_test >= limiar).astype(int)

    #Target used to fit: continuous (default) or discrete (control experiment)
    if discreto:
        y_train_fit = y_train_class
    else:
        y_train_fit = y_train

    #Time series cross validation only inside the train period
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    mae_cv = []
    f1_cv = []
    for idx_train, idx_val in tscv.split(X_train):
        model = create_regressor()
        model.fit(X_train.iloc[idx_train], y_train_fit.iloc[idx_train])
        y_pred_val = model.predict(X_train.iloc[idx_val])
        mae_cv.append(mean_absolute_error(y_train.iloc[idx_val], y_pred_val))
        #Continuous to discrete
        y_pred_val_class = (y_pred_val >= limiar).astype(int)
        f1_cv.append(f1_score(y_train_class.iloc[idx_val], y_pred_val_class, zero_division=0))

    #Final model trained with all the train period
    model = create_regressor()
    model.fit(X_train, y_train_fit)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    #Continuous to discrete
    y_pred_train_class = (y_pred_train >= limiar).astype(int)
    y_pred_test_class = (y_pred_test >= limiar).astype(int)

    #Reference: always predicting the median of the train
    y_pred_baseline = np.full(len(y_test), y_train.median())

    metricas = {
        'target': target,
        'mae_train': round(mean_absolute_error(y_train, y_pred_train)*100, 2),
        'mae_test': round(mean_absolute_error(y_test, y_pred_test)*100, 2),
        'mae_test_baseline': round(mean_absolute_error(y_test, y_pred_baseline)*100, 2),
        'mae_cv': round(np.mean(mae_cv)*100, 2),
        'mae_cv_std': round(np.std(mae_cv)*100, 2),
        'f1_score_train': round(f1_score(y_train_class, y_pred_train_class, zero_division=0), 3),
        'f1_score_test': round(f1_score(y_test_class, y_pred_test_class, zero_division=0), 3),
        'recall_train': round(recall_score(y_train_class, y_pred_train_class, zero_division=0), 3),
        'recall_test': round(recall_score(y_test_class, y_pred_test_class, zero_division=0), 3),
        'precision_train': round(precision_score(y_train_class, y_pred_train_class, zero_division=0), 3),
        'precision_test': round(precision_score(y_test_class, y_pred_test_class, zero_division=0), 3),
        'f1_score_cv': round(np.mean(f1_cv), 3),
        'f1_score_cv_std': round(np.std(f1_cv), 3)
    }

    print('Modelo:', target)
    print('mae_train:', metricas['mae_train'], '%')
    print('mae_test:', metricas['mae_test'], '%')
    print('mae_test_baseline:', metricas['mae_test_baseline'], '%')
    print('mae_cv:', metricas['mae_cv'], '% +/-', metricas['mae_cv_std'])
    print('f1_score_train:', metricas['f1_score_train'])
    print('f1_score_test:', metricas['f1_score_test'])
    print('recall_train:', metricas['recall_train'])
    print('recall_test:', metricas['recall_test'])
    print('precision_train:', metricas['precision_train'])
    print('precision_test:', metricas['precision_test'])
    print('f1_score_cv:', metricas['f1_score_cv'], '+/-', metricas['f1_score_cv_std'])
    print()
    return model, metricas


def train_heads(ds, previsores, modelos, algoritmo=ALGORITMO, limiar=0.5):
    #One model per head that uses a model (the persistence needs no training), saved in models/
    #algoritmo: ALGORITMO of config.py (chosen in the notebook 5.0), class_weight=None
    for minutos in previsores:
        nome = previsores[minutos]
        if nome == 'persistencia':
            print(minutos, 'min: persistencia, nada para treinar')
            continue
        target = 'y_' + str(minutos)
        dx = ds[ds[target].notna()]
        y = (dx[target] >= limiar).astype(int)
        model = create_algorithm(algoritmo)
        model.fit(dx[head_features(modelos[nome], minutos)], y)
        joblib.dump(model, PATH_MODELS / f'modelo_y_{minutos}.pkl')
        print(minutos, 'min:', nome, '(' + algoritmo + ') | modelo salvo: modelo_y_' + str(minutos) + '.pkl | linhas:', len(dx))


def train_heads_cv(ds, features, cabecas, n_splits=5, limiar=0.5):
    #Time series CV of one model (create_model) per head: each fold is predicted only with the past
    #ds: train period with the columns y_15, y_30, ... | features: columns of the model | cabecas: minutes ahead
    #Returns a dict {minutos: table with fold, Data, y_true, persistencia and modelo (probability)}
    resultados = {}
    for minutos in cabecas:
        target = 'y_' + str(minutos)
        dx = ds[ds[target].notna()].sort_values('Data').reset_index(drop=True)
        y = (dx[target] >= limiar).astype(int)
        #Columns x_futuro become x_futuro_<minutos> (solar and climate at the instant t + k)
        colunas = head_features(features, minutos)
        #gap = horizon in rows (5 minutes each): the train never sees a target inside the validation
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=minutos // 5)
        lista = []
        fold = 0
        for idx_train, idx_val in tscv.split(dx):
            #Skipping folds with only one class
            if y.iloc[idx_train].nunique() < 2 or y.iloc[idx_val].sum() == 0:
                continue
            model = create_model(class_weight=None)
            model.fit(dx[colunas].iloc[idx_train], y.iloc[idx_train])
            dy = pd.DataFrame({
                'fold': fold,
                'Data': dx['Data'].iloc[idx_val].values,
                'y_true': y.iloc[idx_val].values,
                'persistencia': dx['y'].iloc[idx_val].values,
                'modelo': model.predict_proba(dx[colunas].iloc[idx_val])[:, 1]
            })
            lista.append(dy)
            fold = fold+1
        resultados[minutos] = pd.concat(lista).reset_index(drop=True)
    return resultados


def out_of_fold(X, alvo, gap, n_blocos=5, limiar=0.5):
    #Prediction of each row by a model trained only with the PAST of the row (time series CV, with gap),
    #like in the validation and in production. The first block has no past: NaN (the LightGBM accepts NaN)
    #alvo: score of the camera (continuous). Rows without alvo (NaN) are not used to train, but are predicted
    previsao = np.full(len(X), np.nan)
    blocos = TimeSeriesSplit(n_splits=n_blocos, gap=gap)
    for idx_fit, idx_pred in blocos.split(X):
        ok = alvo.iloc[idx_fit].notna().values
        y_fit = (alvo.iloc[idx_fit][ok] >= limiar).astype(int)
        if y_fit.nunique() < 2:
            continue
        model = create_model(class_weight=None)
        model.fit(X.iloc[idx_fit][ok], y_fit)
        previsao[idx_pred] = model.predict_proba(X.iloc[idx_pred])[:, 1]
    return previsao


def train_heads_chain_cv(ds, features, cabecas, n_splits=5, limiar=0.5):
    #Chain of heads: the model of the head k also receives the PREDICTIONS of the heads before k
    #(the 60 min head receives the probability of plume predicted for 15 and 30 min: columns prev_15, prev_30)
    #No leak: in each fold the predictions of the previous heads come from models trained only with the train rows,
    #and in the train rows they come from models of the past of each row (out_of_fold), like in the validation
    #Same rows and folds of train_heads_cv, so the two results are compared paired (compare_runs)
    #The first head has no head before it (its "previous prediction" is the persistence, already in score_atual)
    resultados = {}
    for minutos in cabecas:
        anteriores = [cabeca for cabeca in cabecas if cabeca < minutos]
        if len(anteriores) == 0:
            continue
        target = 'y_' + str(minutos)
        dx = ds[ds[target].notna()].sort_values('Data').reset_index(drop=True)
        y = (dx[target] >= limiar).astype(int)
        colunas = head_features(features, minutos)
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=minutos // 5)
        lista = []
        fold = 0
        for idx_train, idx_val in tscv.split(dx):
            if y.iloc[idx_train].nunique() < 2 or y.iloc[idx_val].sum() == 0:
                continue
            X_train = dx[colunas].iloc[idx_train].copy()
            X_val = dx[colunas].iloc[idx_val].copy()
            for anterior in anteriores:
                colunas_anterior = head_features(features, anterior)
                alvo_anterior = dx['y_' + str(anterior)].iloc[idx_train]
                #Train rows: out of fold prediction of the previous head
                X_train['prev_' + str(anterior)] = out_of_fold(dx[colunas_anterior].iloc[idx_train], alvo_anterior, anterior // 5)
                #Validation rows: model of the previous head trained with all the train rows
                ok = alvo_anterior.notna().values
                model = create_model(class_weight=None)
                model.fit(dx[colunas_anterior].iloc[idx_train][ok], (alvo_anterior[ok] >= limiar).astype(int))
                X_val['prev_' + str(anterior)] = model.predict_proba(dx[colunas_anterior].iloc[idx_val])[:, 1]
            model = create_model(class_weight=None)
            model.fit(X_train, y.iloc[idx_train])
            dy = pd.DataFrame({
                'fold': fold,
                'Data': dx['Data'].iloc[idx_val].values,
                'y_true': y.iloc[idx_val].values,
                'persistencia': dx['y'].iloc[idx_val].values,
                'modelo': model.predict_proba(X_val)[:, 1]
            })
            lista.append(dy)
            fold = fold+1
        resultados[minutos] = pd.concat(lista).reset_index(drop=True)
        print(minutos, 'min: cadeia com as previsoes de', anteriores)
    return resultados


def main():
    #Step 3 of the pipeline: processed -> models/modelo_y_<minutos>.pkl of the heads that use a model
    #Trained with ALL the data (train + hold out): the evaluation was already done in the notebooks
    ds = load_processed()
    train_heads(ds, PREVISOR_POR_CABECA, MODELOS)


if __name__ == "__main__":
    main()
