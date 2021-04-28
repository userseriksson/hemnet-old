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
#print('------------------------------------------------')
print('Load data from hemnet')
#print('Done')
#print('------------------------------------------------')

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
#print('------------------------------------------------')
work_date = df.drop(['streetName','broker','Name_brocker','area','date_sold','Prisutveckling','Pris_per_kvadratmeter','newDate','first_streat_name'], axis = 1)
print(work_date.head(1))


# save an part of data in validation 
work_date, validate=predict_functions.train_test_val(work_date,0.9)

#print('-------------------------------------------------')
#print('Validaten have been broken out',validate.shape)
#print('-------------------------------------------------')


# Splitt data in traning and validateion set 
X_train, X_test, y_train, y_test=predict_functions.create_traning_test_data(work_date,0.2)

print('-------------------------------------------------')
print('Set hyperopt space for XG-boost modifications',predict_functions.space)
#print('-------------------------------------------------')

max_evals=800

trials, optparam=predict_functions.hyperopt(work_date,max_evals,save=True)
"""
        Run hyperparameter optimization using the hyperopt package.
        
        processed_data: DataFrame, dataset for one model
        max_evals: upper limit for number of hyperopt trials
        trials: Trials object. To warm start, provide a Trials object from previous training.
            Default is an new object resulting in cold start.
        :return: This trainings Trials, and a dictionary with the optimal hyperparameters
        with the best param save in "best_hyperopt_appartmenet.p"
        """
print(optparam)


result_standard=predict_functions.appartmet_model_standard(work_date)

print('--------Train with out hyperOpt-------------')
print('Traing result',result_standard.head(3))
print('error_kronor',result_standard.error_kronor.mean())
print('error_procent',result_standard.error_procent.mean())



print('-------------------------------------------------')
print('Train the model with the hyper optt')
#print('-------------------------------------------------')


result_hyperOpt=predict_functions.appartmet_model(work_date,True)

print('-------Train with  hyperOpt-------')
print('Traing result HyperOpt',result_hyperOpt.head(3))
print('error_kronor HyperOpt',result_hyperOpt.error_kronor.mean())
print('error_procent HyperOpt',result_hyperOpt.error_procent.mean())




def run_model(work_data,pre_train_model,versions):
    #load an pre train moduel
    if pre_train_model:
        #loaded_model = pickle.load(open("XGBoostHyperOpt.pickle.dat", "rb"))
        loaded_model = pickle.load(open(str(versions), "rb"))
    # Splitt data in X and Y streetName
    X=df.filter(items=['Begärt_pris','Antal_rum','Boarea','Avgift/månad','Driftskostnad','Byggår',
                       'sin_day', 'cos_day', 'sin_week',  'cos_week', 'encode_street', ' encode_brocker_name',
                       'encode_broker',  'encode_area',  'encode_first_streat_name'])
    yhat = loaded_model.predict(X)
    return yhat


print('-------------------------------------- Run on validation data for non turne model --------------------------------------')
#print('Check wit the validation data ',result_hyperOpt.head(3))

# function for the model 
yhat=run_model(validate,True,versions='XGBoostStandard.pickle.dat')

data_yhat_pred = pd.DataFrame(yhat)
data_yhat_pred = data_yhat_pred.reset_index(drop=True)
data_validation = pd.DataFrame(validate)
data_validation = data_validation.reset_index(drop=True)
final = pd.concat([data_validation, data_yhat_pred], axis=1)
final = final[['price','Begärt_pris','Antal_rum','Boarea',0]]
final.columns = ['price','Begärt_pris','Antal_rum','Boarea','Yhat']

final['error_kronor'] = ((final['price']-final['Yhat']))
final['error_procent'] = (abs(final['price']-final['Yhat'])/final['Yhat'])
final['price_per_square'] = (final['Yhat']/final['Boarea'])
print('error_kronor',final['error_kronor'].mean())
print('error_procent',final['error_procent'].mean())
print('price_per_square',final['price_per_square'].mean())
print(final.head(5))




#print(yhat)

print('--------------------------------------Run on validation data with hyper OptModel --------------------------------------')

yhat=run_model(validate,True,versions='XGBoostHyperOpt.pickle.dat')

data_yhat_pred = pd.DataFrame(yhat)
data_yhat_pred = data_yhat_pred.reset_index(drop=True)
data_validation = pd.DataFrame(validate)
data_validation = data_validation.reset_index(drop=True)
final = pd.concat([data_validation, data_yhat_pred], axis=1)
final = final[['price','Begärt_pris','Antal_rum','Boarea',0]]
final.columns = ['price','Begärt_pris','Antal_rum','Boarea','Yhat']
#final = df[['price','Begärt_pris','Antal_rum','Boarea',0]]
#print(final.head(1))
final['error_kronor'] = ((final['price']-final['Yhat']))
final['error_procent'] = (abs(final['price']-final['Yhat'])/final['Yhat'])
final['price_per_square'] = (final['Yhat']/final['Boarea'])
print('error_kronor',final['error_kronor'].mean())
print('error_procent',final['error_procent'].mean())
print('price_per_square',final['price_per_square'].mean())
print(final.head(5))



'''
final.to_csv('/Users/joeriksson/Desktop/python_data/vasa_kungsholmen_pred_20210426.csv')


'''