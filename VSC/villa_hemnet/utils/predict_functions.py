import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def create_traning_test_data(df,size):
    # Splitt data in X and Y streetName
    X=df.filter(items=['Begärt_pris','Antal_rum','Boarea','Avgift/månad','Driftskostnad','Byggår',
                       'sin_day', 'cos_day', 'sin_week',  'cos_week', 'encode_street', ' encode_brocker_name', 'encode_broker',  'encode_area',  'encode_first_streat_name'
                       ])
    #target data.
    y=df.filter(items=['price'])
    data_dmatrix = xgb.DMatrix(data=X,label=y, enable_categorical=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=1234567)
    #print('X_train',X_train.shape)
    #print('X_test',X_test.shape)
    #print('y_train',y_train.shape)
    #print('y_test',y_test.shape)
    return X_train, X_test, y_train, y_test


space={'max_depth': hp.quniform("max_depth", 3, 10, 1),
       'gamma': hp.uniform('gamma',0,3),
       'reg_alpha' : hp.quniform('reg_alpha', 0,90,1),
       'reg_lambda' : hp.uniform('reg_lambda', 0,40),
       'colsample_bytree' : hp.uniform('colsample_bytree', 0.1,1),
       'min_child_weight' : hp.quniform('min_child_weight', 0, 16, 1),
       'n_estimators': hp.quniform('n_estimators', 300, 3000, 1),
       'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1.0)),
       #'subsample': hp.uniform('subsample', 0.5, 1.0),
       'early_stopping_rounds': hp.quniform('early_stopping_rounds', 250, 500, 1)}


def hyperparameter_tuning(space,df):
    reg=xgb.XGBRegressor(max_depth = int(space['max_depth']),
                         gamma = space['gamma'],
                         reg_alpha = int(space['reg_alpha']),
                         reg_lambda = space['reg_lambda'],
                         colsample_bytree = space['colsample_bytree'],
                         min_child_weight = int(space['min_child_weight']),
                         n_estimators = int(space['n_estimators']),
                         learning_rate = space['learning_rate'],
                         subsample = space['subsample'],
                         objective = 'reg:squarederror')

    X_train, X_test, y_train, y_test = create_traning_test_data(df,0.2)
    X_train=X_train.fillna(0)
    y_test=y_test.fillna(0)
    X_train=X_train.fillna(0)
    y_train=y_train.fillna(0)
    evaluation = [( X_train, y_train), ( X_test, y_test)]

    reg.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=400,verbose=False)

    pred = reg.predict(X_test)
    mse= mean_squared_error(y_test, pred)
    rmse=np.sqrt(mean_squared_error( y_test,pred))
    #change the metric if you like
    return {'loss':rmse, 'status': STATUS_OK }



def run_optimaziation(hyperparameter_tuning,space,evals,save):
    trials = Trials()
    best = fmin(fn=hyperparameter_tuning,
                space=space,
                algo=tpe.suggest,
                max_evals=evals,
                trials=trials)
    print(best)
    if save:
        pickle.dump(best, open("best_param_xgb.p", "wb"))

    return best