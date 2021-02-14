import pandas as pd
import numpy as np
import altair as alt
import pickle

import streamlit as st
import altair as alt
from vega_datasets import data
import time
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#plt.style.use('fivethirtyeight')


def run_model(df):
    #load moduel
    loaded_model = pickle.load(open("xg_opti.pickle.dat", "rb"))
    # Splitt data in X and Y streetName
    X=df.filter(items=['encode_street','encode_area','Begärt_pris','encode_name','Antal_rum','Boarea','avgift','Driftskostnad','byggår','dayofweek','quater','month','dayofyear'])
    yhat = loaded_model.predict(X)
    return yhat



#read an plk data
k = pd.read_pickle('apartment_data.plk') #to load 123.pkl back to the dataframe df
df=k
df.columns = df.columns.str.replace(' ', '_')
df = k.drop(['info','Prisutveckling'], axis = 1)
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
df['broker']=df['broker'].str.replace(' ','')
df['encode_area'] = lbl.fit_transform(df['area'].fillna('').astype(str))

print(df.head())

print(df.dtypes)

#print(df.groupby(['info']).count())

print(df.Name_brocker.unique())

d = pd.date_range(start='9/01/2020', end='2/01/2021', freq='D')
d = pd.DataFrame(d)
d.columns=['date']

#work_data_appartmet=pd.merge(d, df, how='left',left_on=d[0], right_on=df['date_sold'])


import seaborn as sns
import streamlit as st
import pandas as pd


stock_data = pd.read_csv('/Users/joeriksson/Desktop/python_data/swe_test_stock.csv',sep = ',')
actual_stock_data = pd.read_csv('/Users/joeriksson/Desktop/python_data/stock_OMX_20201120.csv', sep = ',')

st.title('Here you can find OMX stock, ABB, Aztra, Alfa Laval')
st.subheader('Prediction of stock upcoming 180 days')
#st.sidebar.checkbox("Show Analysis of stock", True, key=1)
#select = st.sidebar.selectbox('Select a stock',list(set(stock_data['name'])))



# Load moduel

loaded_model = pickle.load(open("xg_opti.pickle.dat", "rb"))

#---------- LINE plott ----------
star_date = '2021-05-01'
end_date = '2019-01-01'
#stock_data['ds'] = pd.to_datetime(stock_data['ds'], format='%Y-%m-%d')
#actual_stock_data['ds']=actual_stock_data['Date'] = pd.to_datetime(actual_stock_data['Date'], format='%Y-%m-%d')






#---------- Line plott predict data -----------
# Plott data with end prices
end_price = df.filter(items=['date_sold','area','price','Boarea','ppk'])
end_price=pd.merge(d, end_price, how='left',left_on=d['date'], right_on=df['date_sold'])
end_price.columns=['date','second_date','date_sold','area','price','Boarea','ppk']

time=end_price.groupby(['date','area'])['ppk'].mean().reset_index()
time=(pd.DataFrame(time))
#time = time.set_index('date')


#st.line_chart(time)
#time = time.set_index(end_price['date'])
basic_chart = alt.Chart(time).mark_line().encode(
    x='date',
    y='ppk',
    #color='area',
    # legend=alt.Legend(title='Animals by year')
)
st.altair_chart(basic_chart)


print(df.info())
st.write("Here's data with actual and pred :")
st.write(pd.DataFrame(time))
st.write(pd.DataFrame(df))
#st.write(pd.DataFrame(end_price_v))

yhat=run_model(df.iloc[-1:])
st.write(df.iloc[-1:])
st.write(pd.DataFrame(yhat.round(0)))

st.write("Here's data with actual and pred :")
st.write(df.iloc[-5:])

import streamlit as st
import altair as alt
from vega_datasets import data
import time


data = pd.DataFrame({
    'city': ['Cincinnati', 'San Francisco', 'Pittsburgh'],
    'sports_teams': [6, 8, 9],
})

st.write(df)
F=st.write(alt.Chart(df).mark_bar().encode(
    x=alt.X('area', sort=None),
    y='price',
))
st.bar_chart(f)


'''#---------- Line plott predict data -----------
end_price_v = end_price
st.write(pd.DataFrame(end_price))
mylist = list(set(end_price['area']))
option = st.selectbox('Select area?',(mylist))
st.write('You selected:', option)
name=option
st.subheader("Totall price per area")
end_price = end_price[end_price['area']==name]
chart_data_v2 = pd.DataFrame(end_price['ppk'])
chart_data_v2 = chart_data_v2.set_index(end_price['date_sold'])
st.line_chart(chart_data_v2)'''