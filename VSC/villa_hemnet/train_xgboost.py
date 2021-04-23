from utils import help_functions
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#read an plk data
old = pd.read_pickle('vasa_kungsholmen.plk') #to load 123.pkl back to the dataframe df
df=old
df.columns = df.columns.str.replace(' ', '_')
df = old.drop(['info','broker','Name_brocker','Prisutveckling'], axis = 1)

print(df.head())
print(df.tail())

help_functions.help(df)

print("--------------------------------")
#print(df.head())