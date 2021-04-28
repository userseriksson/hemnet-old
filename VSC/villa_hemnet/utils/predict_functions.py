import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error for y_true and y_pred
    """
    err = None
    if not (y_true or y_pred):
        err = sqrt(mean_squared_error(y_true.to_numpy(), y_pred.to_numpy()))
    return err

def train_test_val(df,splitt_kvote):
    train_test, validate = np.split(df.sample(frac=1), [int(splitt_kvote*len(df))])
    return train_test, validate


def create_traning_test_data(df,size):
    # Splitt data in X and Y streetName
    X=df.filter(items=['Begärt_pris','Antal_rum','Boarea','Avgift/månad','Driftskostnad','Byggår',
                       'sin_day', 'cos_day', 'sin_week',  'cos_week', 'encode_street', ' encode_brocker_name', 'encode_broker',  'encode_area',  'encode_first_streat_name'
                       ])
    #target data.
    y=df.filter(items=['price'])
    data_dmatrix = xgb.DMatrix(data=X,label=y, enable_categorical=True)
    # Create traing and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=1234567)
    #print('X_train',X_train.shape)
    #print('X_test',X_test.shape)
    #print('y_train',y_train.shape)
    #print('y_test',y_test.shape)
    return X_train, X_test, y_train, y_test

#'''
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
#'''

def hyperopt(processed_data, max_evals, save, trials=Trials()):
        """
        Run hyperparameter optimization using the hyperopt package.
        
        processed_data: DataFrame, dataset for one model
        max_evals: upper limit for number of hyperopt trials
        trials: Trials object. To warm start, provide a Trials object from previous training.
            Default is an new object resulting in cold start.
        :return: This trainings Trials, and a dictionary with the optimal hyperparameters
        """
        X_train, X_test, y_train, y_test=create_traning_test_data(processed_data,0.2)
        
        def objective(space):
            """
            Train a regressor using the hyperparameters in space and return the best loss.
            """

            reg = xgb.XGBRegressor(max_depth = int(space['max_depth']),
                                                   gamma = space['gamma'],
                                                   reg_alpha = int(space['reg_alpha']),
                                                   reg_lambda = space['reg_lambda'],
                                                   colsample_bytree = space['colsample_bytree'],
                                                   min_child_weight = int(space['min_child_weight']),
                                                   n_estimators = int(space['n_estimators']),
                                                   learning_rate = space['learning_rate'],
                                                   #subsample = space['subsample'],
                                                   objective = 'reg:squarederror')
            
            evaluation = [( X_train, y_train), ( X_test, y_test)]
            
            reg.fit(X_train, y_train,
                    eval_set=evaluation, eval_metric="rmse",
                    early_stopping_rounds=100000,verbose=False)

            pred = reg.predict(X_test)
            mse= mean_squared_error(y_test, pred)
            rmse=np.sqrt(mean_squared_error( y_test,pred))
            #change the metric if you like
            return {'loss':rmse, 'status': STATUS_OK }
            
        optparam = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials)
        print(optparam)
        if save:
            pickle.dump(optparam, open("best_hyperopt_appartmenet.p", "wb"))
        return trials, optparam


def appartmet_model_standard(df):
    
    #create tring data
    X_train, X_test, y_train, y_test = create_traning_test_data(df,0.2)
    
    xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror')
    
    xgb_model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      early_stopping_rounds=100000,
                      verbose=False)
    y_pred = xgb_model.predict(X_test)
    
    #crate data set 
    loss=np.sqrt(mean_squared_error(y_test,y_pred))
    print(f'loss {loss}')
    
    #save standard model 
    pickle.dump(xgb_model, open("XGBoostStandard.pickle.dat", "wb"))

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
    #print(result)
    #result['RSME']=np.sqrt(mean_squared_error(result['price'],result['pred_price']))
    result['error_kronor'] = ((result['price']-result['pred_price']))
    result['error_procent'] = (abs(result['price']-result['pred_price'])/result['pred_price'])
    result['error_on_end_price'] = (abs(result['error_kronor']/result['price']))
    #print('RSME',result['RSME'].mean())
    print('error_kronor',result['error_kronor'].mean())
    print('error_procent',result['error_procent'].mean())
    return result
    

# trin the actual model 

def appartmet_model(df,best):
    if best:
        best = pickle.load(open("best_param_xgb.p", "rb"))
    #print(f'Setup models for {best.keys()}')

    X_train, X_test, y_train, y_test = create_traning_test_data(df,0.2)
    #fit moddel
    
    print(f'Train input shape {X_train.shape}')
            
    mdl = (xgb.XGBRegressor(max_depth = int(best['max_depth']),
                                     gamma = best['gamma'],
                                     reg_alpha = int(best['reg_alpha']),
                                     reg_lambda = best['reg_lambda'],
                                     colsample_bytree = best['colsample_bytree'],
                                     min_child_weight = int(best['min_child_weight']),
                                     n_estimators = int(best['n_estimators']),
                                     learning_rate = best['learning_rate'],
                                     #subsample = int(best['subsample']),
                                     objective = 'reg:squarederror'))
            
    mdl.fit(X_train, y_train,
                            eval_set=[(X_train, y_train), (X_test, y_test)],
                            early_stopping_rounds=100000,
                            #eval_metric=['mae', 'rmse'],
                            #allow_negatives=False
                            verbose=False
                            )

    y_pred = mdl.predict(X_test) 
    #loss = rmse(y_test, y_pred)
    loss=np.sqrt(mean_squared_error(y_test,y_pred))
    print(f'loss {loss}')
    
    # Save model
    pickle.dump(mdl, open("XGBoostHyperOpt.pickle.dat", "wb"))

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
    #print(result)
    #result['RSME']=np.sqrt(mean_squared_error(result['price'],result['pred_price']))
    result['error_kronor'] = ((result['price']-result['pred_price']))
    result['error_procent'] = (abs(result['price']-result['pred_price'])/result['pred_price'])
    result['error_on_end_price'] = (abs(result['error_kronor']/result['price']))
    #print('RSME',result['RSME'].mean())
    print('error_kronor',result['error_kronor'].mean())
    print('error_procent',result['error_procent'].mean())
    return  result

    
    
    
    
    
    
    
    
    '''
    

    xgb_model_opt.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      early_stopping_rounds=10000,
                      verbose=False)

    # calculate the auc score
    y_pred = xgb_model_opt.predict(X_test)
    
    # save model to file
    pickle.dump(xgb_model_opt, open("xg_opti.pickle.dat", "wb"))
    #print(xgb_model_opt)

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
    #print(result)
    #result['RSME']=np.sqrt(mean_squared_error(result['price'],result['pred_price']))
    result['error_kronor'] = ((result['price']-result['pred_price']))
    result['error_procent'] = (abs(result['price']-result['pred_price'])/result['pred_price'])
    result['error_on_end_price'] = (abs(result['error_kronor']/result['price']))
    #print('RSME',result['RSME'].mean())
    print('error_kronor',result['error_kronor'].mean())
    print('error_procent',result['error_procent'].mean())
    return  result



def hyperparameter_tuning(space, X_train, X_test, y_train, y_test):
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

'''