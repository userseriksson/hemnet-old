from utils import help_functions
from utils import predict_functions
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# if error to acitvate eval "$(pyenv init -)"

#read an plk data
old = pd.read_pickle('vasa_kungsholmen.plk') #to load 123.pkl back to the dataframe df
df=old
df.columns = df.columns.str.replace(' ', '_')
#df = old.drop(['info','broker','Name_brocker','Prisutveckling'], axis = 1)
df = old.drop(['info','Biarea'], axis = 1)
print('------------------------------------------------')
print('Load data from hemnet')
print('Done')
print('------------------------------------------------')

#print(df.head())
#print(df.tail())

#print("--------------------------------")
df=help_functions.categry_name(df)
#print("--------------------------------")

# Extract the solde date 
df=help_functions.mdy_to_ymd(df)
#print("--------------------------------")
df=help_functions.uniqe_streat_name(df,'streetName')
#print("--------------------------------")
df=help_functions.cyclical_transform_day(df)
#print("--------------------------------")
df=help_functions.cyclical_transform_week(df)
#print("--------------------------------")
df=help_functions.encoding(df)
#print(df.info())

print('------------------------------------------------')
print('Pre proccsing done')
print('------------------------------------------------')
work_date = df.drop(['streetName','broker','Name_brocker','area','date_sold','Prisutveckling','Pris_per_kvadratmeter','newDate','first_streat_name'], axis = 1)
print(work_date.head(1))


# Splitt data in traning and validateion set 
#X_train, X_test, y_train, y_test=predict_functions.create_traning_test_data(df,0.2)

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

    X_train, X_test, y_train, y_test = predict_functions.create_traning_test_data(df,0.2)
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

run_optimaziation(hyperparameter_tuning,space,350,True)

def appartmet_model(df,best):
    if best:
        best = pickle.load(open("best_param_xgb.p", "rb"))

    X_train, X_test, y_train, y_test = predict_functions.create_traning_test_data(df,0.2)
    #fit moddel
    xgb_model_opt = xgb.XGBRegressor(max_depth = int(best['max_depth']),
                                     gamma = best['gamma'],
                                     reg_alpha = int(best['reg_alpha']),
                                     reg_lambda = best['reg_lambda'],
                                     colsample_bytree = best['colsample_bytree'],
                                     min_child_weight = int(best['min_child_weight']),
                                     n_estimators = int(best['n_estimators']),
                                     learning_rate = best['learning_rate'],
                                     #subsample = int(best['subsample']),
                                     objective = 'reg:squarederror')

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
    #print(data_feuteres)

    # save model to file
    pickle.dump(xgb_model_opt, open("xg_opti.pickle.dat", "wb"))
    print(xgb_model_opt)

    #y_pred = xgb_model.predict(X)
    data_pred = pd.DataFrame(y_pred)
    data_pred = data_pred.reset_index(drop=True)
    data_ytest = pd.DataFrame(y_test)
    data_ytest = data_ytest.reset_index(drop=True)
    # add all data 
    data_xtest = pd.DataFrame(X_test)
    data_xtest = data_xtest.reset_index(drop=True)

    result = pd.concat([ data_ytest, data_pred], axis=1)
    result.columns = ['price', 'pred_price']
    result = pd.concat([ data_xtest, result], axis=1)
    print(result)
    #result['RSME']=np.sqrt(mean_squared_error(result['price'],result['pred_price']))
    result['error_kronor'] = ((result['price']-result['pred_price']))
    result['error_procent'] = (abs(result['price']-result['pred_price'])/result['pred_price'])
    result['error_on_end_price'] = (abs(result['error_kronor']/result['price']))
    #print('RSME',result['RSME'].mean())
    print('error_kronor',result['error_kronor'].mean())
    print('error_procent',result['error_procent'].mean())
    return  result,feature_important

result,data_feuteres=appartmet_model(df,True)


print(result.head(10))
print('---------------------')
print(data_feuteres)


def run_model(df,pre_train_model):
    #load an pre train moduel
    if pre_train_model:
        loaded_model = pickle.load(open("xg_opti.pickle.dat", "rb"))
    # Splitt data in X and Y streetName
    X=df.filter(items=['Begärt_pris','Antal_rum','Boarea','Avgift/månad','Driftskostnad','Byggår',
                       'sin_day', 'cos_day', 'sin_week',  'cos_week', 'encode_street', ' encode_brocker_name',
                       'encode_broker',  'encode_area',  'encode_first_streat_name'])
    yhat = loaded_model.predict(X)
    return yhat

yhat=run_model(df,True)
print(yhat)

data_yhat_pred = pd.DataFrame(yhat)
data_yhat_pred = data_yhat_pred.reset_index(drop=True)
data_validation = pd.DataFrame(df)
data_validation = data_validation.reset_index(drop=True)
final = pd.concat([data_validation, data_yhat_pred], axis=1)
print(final)

final.to_csv('/Users/joeriksson/Desktop/python_data/vasa_kungsholmen_pred_20210426.csv')


