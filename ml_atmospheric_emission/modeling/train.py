import joblib
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from ml_atmospheric_emission.config import FEATURES, PATH_MODELS


def create_model(class_weight='balanced'):
    #Tree model with conservative parameters to avoid overfitting
    #class_weight=None gives probabilities closer to the real chance of smog (used to recommend)
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight=class_weight,
        random_state=42,
        verbose=-1
    )
    return model


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


def train_heads(ds, previsores, modelos, limiar=0.5):
    #One model per head that uses a model (the persistence needs no training), saved in models/
    #class_weight=None: the output is closer to the real probability of plume
    for minutos in previsores:
        nome = previsores[minutos]
        if nome == 'persistencia':
            print(minutos, 'min: persistencia, nada para treinar')
            continue
        target = 'y_' + str(minutos)
        dx = ds[ds[target].notna()]
        y = (dx[target] >= limiar).astype(int)
        model = create_model(class_weight=None)
        model.fit(dx[modelos[nome]], y)
        joblib.dump(model, PATH_MODELS / f'modelo_y_{minutos}.pkl')
        print(minutos, 'min:', nome, '| modelo salvo: modelo_y_' + str(minutos) + '.pkl | linhas:', len(dx))
