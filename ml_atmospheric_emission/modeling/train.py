import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from ml_atmospheric_emission.config import FEATURES


def create_model():import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from ml_atmospheric_emission.config import FEATURES


def create_model():
    #Tree model with conservative parameters to avoid overfitting
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
        class_weight='balanced',
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



def train_model_experiment(ds, target, features=FEATURES, test_size=0.2, n_splits=5, gap=0):
    ds = ds.copy()
    #Sorting by time and removing rows without target
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds[ds[target].notna()].reset_index(drop=True)
    X = ds[features]
    y = ds[target].astype(int)

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

    print('Modelo:', target)
    print('f1_score_train:', round(f1_score(y_train, y_pred_train, zero_division=0), 3))
    print('f1_score_test:', round(f1_score(y_test, y_pred_test, zero_division=0), 3))
    print('recall_train:', round(recall_score(y_train, y_pred_train, zero_division=0), 3))
    print('recall_test:', round(recall_score(y_test, y_pred_test, zero_division=0), 3))
    print('precision_train:', round(precision_score(y_train, y_pred_train, zero_division=0), 3))
    print('precision_test:', round(precision_score(y_test, y_pred_test, zero_division=0), 3))
    print('f1_score_cv:', round(np.mean(f1_cv), 3), '+/-', round(np.std(f1_cv), 3))
    print('')
    return model


def train_model_experiment(ds, target, features=FEATURES, test_size=0.2, n_splits=5, gap=0):
    ds = ds.copy()
    #Sorting by time and removing rows without target
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds[ds[target].notna()].reset_index(drop=True)
    X = ds[features]
    y = ds[target].astype(int)

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

    print('Modelo:', target)
    print('f1_score_train:', round(f1_score(y_train, y_pred_train, zero_division=0), 3))
    print('f1_score_test:', round(f1_score(y_test, y_pred_test, zero_division=0), 3))
    print('recall_train:', round(recall_score(y_train, y_pred_train, zero_division=0), 3))
    print('recall_test:', round(recall_score(y_test, y_pred_test, zero_division=0), 3))
    print('precision_train:', round(precision_score(y_train, y_pred_train, zero_division=0), 3))
    print('precision_test:', round(precision_score(y_test, y_pred_test, zero_division=0), 3))
    print('f1_score_cv:', round(np.mean(f1_cv), 3), '+/-', round(np.std(f1_cv), 3))
    print('')
    return model