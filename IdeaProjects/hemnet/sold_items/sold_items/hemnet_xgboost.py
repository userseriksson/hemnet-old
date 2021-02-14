import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#read an plk data
k = pd.read_pickle('old_items.plk') #to load 123.pkl back to the dataframe df

df=k
df.columns = df.columns.str.replace(' ', '_')
df = k.drop(['info','broker','Name_brocker','Prisutveckling'], axis = 1)

'''
function to clean data from scraper.
    Args:
        df: Data from scraper
'''
def rename_cloums(df):
    df['streetName']=df['streetName'].astype(str)
    df['price']=df['price'].astype(str).astype(float)
    df['Begärt_pris']=df['Begärt_pris'].astype(str).astype(float)
    df['ppk']=df['Pris_per_kvadratmeter'].astype(str).astype(float)
    df['Antal_rum']=df['Antal_rum'].str.replace('rum','')
    df['Antal_rum']=df['Antal_rum'].str.replace(',','.')
    df['Boarea']=df['Boarea'].str.replace(',','.')
    df['date_sold']=df['date_sold'].str.replace(' ','')
    df['date_sold']=df['date_sold'].str.replace('januari','-jan-')
    df['date_sold']=df['date_sold'].str.replace('february','-feb-')
    df['date_sold']=df['date_sold'].str.replace('december','-dec-')
    df['date_sold']=df['date_sold'].str.replace('november','-nov-')
    df['date_sold']=df['date_sold'].str.replace('oktober','-oct-')
    df['date_sold'] = pd.to_datetime(df['date_sold'])
    df['Driftskostnad'].astype(str).astype(float)
    df['Byggår']=df['Byggår'].str.replace('-','')
    df['Byggår'].astype(str)
    df=df.drop(['Pris_per_kvadratmeter'], axis = 1)
    #df['Prisutveckling'] =  pd.to_numeric(df['Prisutveckling'], errors='coerce')
    df['Antal_rum'] = df['Antal_rum'].astype(float)
    df['Boarea'] = df['Boarea'].astype(float)
    df['avgift'] = pd.to_numeric(df['Avgift/månad'], errors='coerce')
    df['Driftskostnad'] = df['Driftskostnad'].astype(float)
    df['Byggår'] = df['Byggår'].astype(float)
    df['ppk'] = df['ppk'].astype(float)
    df['dayofweek'] = df['date_sold'].dt.dayofweek
    df['quarter'] = df['date_sold'].dt.quarter
    df['month'] = df['date_sold'].dt.month
    #df['year'] = df['date_sold'].dt.year
    df['dayofyear'] = df['date_sold'].dt.dayofyear
    #df['dayofmonth'] = df['date_sold'].dt.day
    lbl = preprocessing.LabelEncoder()
    df['encode_street'] = lbl.fit_transform(df['streetName'].fillna('').astype(str))
    df.loc[df['area'].str.contains('Såld den'), 'area'] = 'NA'
    df['area']=df['area'].str.replace(' ','')
    df['area']=df['area'].str.replace('-','/')
    df['encode_area'] = lbl.fit_transform(df['area'].fillna('').astype(str))
    return df
df=rename_cloums(df)

'''
Create traing test data. use inside optimze and traning functions 
    Args:
        df: Data from scraper
        size: splitt betewwn train/test, ex 0.2
'''
def create_traning_test_data(df,size):
    # Splitt data in X and Y streetName
    X=df.filter(items=['encode_street','encode_area','Begärt_pris','encode_name','Antal_rum','Boarea','avgift',
                       'Driftskostnad','byggår','dayofweek','quater','month','dayofyear'])
    #target data.
    y=df.filter(items=['price'])
    data_dmatrix = xgb.DMatrix(data=X,label=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=1234567)
    return X_train, X_test, y_train, y_test


'''
Definde grid for optimations serche  
'''
space={'max_depth': hp.quniform("max_depth", 3, 10, 1),
       'gamma': hp.uniform('gamma',0,3),
       'reg_alpha' : hp.quniform('reg_alpha', 0,90,1),
       'reg_lambda' : hp.uniform('reg_lambda', 0,40),
       'colsample_bytree' : hp.uniform('colsample_bytree', 0.1,1),
       'min_child_weight' : hp.quniform('min_child_weight', 0, 16, 1),
       'n_estimators': hp.quniform('n_estimators', 300, 3000, 1),
       'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(1.0)),
       'subsample': hp.uniform('subsample', 0.5, 1.0),
       'early_stopping_rounds': hp.quniform('early_stopping_rounds', 250, 500, 1)}

#######################################
#pre_train hyper optimation for xgboost
#######################################

'''
Prepper xg-boost omtpimatxzion for apparment data
    Args:
        df:DataFarme
        space: grid of hypoeroppt parameters
'''
def hyperparameter_tuning(space):
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
#######################################
#optimation for xgboost
#######################################
'''
function to retring wight of the hyperoptimatsion 
    Args:
        hyperparameter_tuning: fuction to optimize
        space: grid of hypoeroppt parameters
        evals: number of runs
        save: False/True statet what to save new params
'''
#best=run_optimaziation(hyperparameter_tuning,space,2,False)


'''
Train XG-Boost modeil with optimazet wights from histroul data set
    Arg:
        df: dataset what to predict on 
        best: True if what to use best params
'''

def appartmet_model(df,best):
    if best:
        best = pickle.load(open("best_param_xgb.p", "rb"))

    X_train, X_test, y_train, y_test = create_traning_test_data(df,0.2)
    #fit moddel
    xgb_model_opt = xgb.XGBRegressor(objective='reg:squarederror',
                           learning_rate =best['learning_rate'],
                           n_estimators =int(best['n_estimators']),
                           max_depth = int(best['max_depth']),
                           gamma = best['gamma'],
                           reg_alpha = int(best['reg_alpha']),
                           min_child_weight=best['min_child_weight'],
                           colsample_bytree=best['colsample_bytree'])
    xgb_model_opt.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=10000,
            verbose=False)

    # calculate the auc score
    y_pred = xgb_model_opt.predict(X_test)
    feature_important = xgb_model_opt.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    data_feuteres = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    print(data_feuteres)

    # save model to file
    pickle.dump(xgb_model_opt, open("xg_opti.pickle.dat", "wb"))
    print(xgb_model_opt)

    #y_pred = xgb_model.predict(X)
    data_pred = pd.DataFrame(y_pred)
    data_pred = data_pred.reset_index(drop=True)
    data_ytest = pd.DataFrame(y_test)
    data_ytest = data_ytest.reset_index(drop=True)

    result = pd.concat([data_ytest, data_pred], axis=1)
    result.columns = ['price', 'pred_price']
    result['RSME']=np.sqrt(mean_squared_error(result['price'],result['pred_price']))
    result['error'] = (abs(result['price']-result['pred_price'])/result['pred_price'])
    print(result['RSME'].mean())
    print(result['error'].mean())

    return result['error'].mean(), data_feuteres

error,data_feuteres=appartmet_model(df,True)
print('Error',error)
print('After',data_feuteres)


'''
run train moduel on DataFrame 
    Arg:
        df: dataset what to predict on 
        pre_train_model: Model from prives train model
'''

def run_model(df,pre_train_model):
    #load an pre train moduel
    if pre_train_model:
        loaded_model = pickle.load(open("xg_opti.pickle.dat", "rb"))
    # Splitt data in X and Y streetName
    X=df.filter(items=['encode_street','encode_area','Begärt_pris','encode_name','Antal_rum','Boarea','avgift',
                       'Driftskostnad','byggår','dayofweek','quater','month','dayofyear'])
    yhat = loaded_model.predict(X)
    return yhat


yhat=run_model(df.iloc[-1:],True)
print(df.iloc[-1:])
print(yhat)