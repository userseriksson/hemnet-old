import seaborn as sns
import streamlit as st
import pandas as pd


stock_data = pd.read_csv('/Users/joeriksson/Desktop/python_data/swe_test_stock.csv',sep = ',')
actual_stock_data = pd.read_csv('/Users/joeriksson/Desktop/python_data/stock_OMX_20201120.csv', sep = ',')


st.title('Here you can find OMX stock, ABB, Aztra, Alfa Laval')
st.subheader('Prediction of stock upcoming 180 days')
#st.sidebar.checkbox("Show Analysis of stock", True, key=1)
#select = st.sidebar.selectbox('Select a stock',list(set(stock_data['name'])))


#---------- LINE plott ----------
star_date = '2021-05-01'
end_date = '2019-01-01'
stock_data['ds'] = pd.to_datetime(stock_data['ds'], format='%Y-%m-%d')
actual_stock_data['ds']=actual_stock_data['Date'] = pd.to_datetime(actual_stock_data['Date'], format='%Y-%m-%d')

#---------- Line plott predict data -----------
mylist = list(set(stock_data['name']))
option = st.selectbox('Select stock are you interested in?',(mylist))
st.write('You selected:', option)
name=option
st.subheader("Prediction of "+name+ " untill 2021-05-01")
work_data = stock_data[stock_data['name']==name]
chart_data_v2 = pd.DataFrame(work_data['yhat'])
chart_data_v2 = chart_data_v2.set_index(work_data['ds'])
st.line_chart(chart_data_v2)

#--------- Plot actual data -----------
st.subheader('Actual stock price history')
mylist_actual = list(set(actual_stock_data['name']))
option_actual = st.selectbox('Select stock are you interested in?',(mylist_actual))
st.write('You selected:', option_actual)
name_actual=option_actual
st.subheader("Actual data for stock "+name_actual)
work_data_actual = actual_stock_data[actual_stock_data['name']==name]
chart_data_actual = pd.DataFrame(work_data_actual['Close'])
chart_data_actual = chart_data_actual.set_index(work_data_actual['Date'])
st.line_chart(chart_data_actual)

#--------------------
#st.write("Here's table of all data:")
#st.write(pd.DataFrame(stock_data))

#---------- Plot actual data
#st.write("Here's table of actaul data:")
#st.write(pd.DataFrame(actual_stock_data))

actuel=actual_stock_data[['ds','name','Close']]
pred=stock_data[['ds','name','yhat']]

merged = pd.merge(actuel, pred, on=["ds", "name"], how="left")
merged_filter=merged.loc[merged["ds"].between(end_date, star_date)]
merged_filter['error'] = (abs(merged_filter['Close']-merged_filter['yhat']))/merged_filter['Close']
#merged_filter['RSME'] =  (np.sqrt(mean_squared_error(merged_filter['Close'],merged_filter['yhat']))
#                          /(merged_filter['Close'])+merged_filter['yhat'])

st.write("Here's data with actual and pred :")
st.write(pd.DataFrame(merged_filter))


#fig = px.line(merged_filter, x='ds', y=['Close', 'yhat'])
# Show plot
#st.line_chart(fig)

#--------- Plot actual data aand predictid data  -----------
#mylist_actual_pred = list(set(merged_filter['name']))
#option_actual_pred = st.selectbox('stock are you interested in?',(mylist_actual_pred))
#st.write('You selected - new:', option_actual_pred)
#name_actual_pred=option_actual_pred
##st.subheader("Prediction of "+name_actual_pred+ " untill 2021-05-01")
#work_data_actual_pred = merged_filter[merged_filter['name']==name]
#chart_data_actual_pred = px.line(merged_filter, x='ds', y=['Close', 'yhat'])
#chart_data_actual_pred = chart_data_actual_pred.set_index(work_data_actual_pred['ds'])
#st.line_chart(chart_data_actual_pred)
#st.line_chart(chart_data_actual_pred)




