import pandas as pd
import numpy as np
import re
from sklearn import preprocessing

def help(df):
    print('Help ------------------------------',df.head())
    
def categry_name(df):
    """
    Transform to right format
    
    df: dataframe with all colums 
    """
    df['streetName']=df['streetName'].astype(str)
    df['price']=df['price'].astype(float)
    df['Prisutveckling']=df['Prisutveckling'].astype(str)
    df['Avgift/månad']=df['Avgift/månad'].astype(str)
    df['Driftskostnad']=df['Driftskostnad'].astype(float)
    df['Byggår']=df['Byggår'].astype(float)
    # Antal rum
    df['Antal_rum']=df['Antal_rum'].str.replace('rum','')
    df['Antal_rum']=df['Antal_rum'].str.replace(' ','')
    df['Antal_rum']=df['Antal_rum'].str.replace(',','.')
    df['Antal_rum'] = pd.to_numeric(df['Antal_rum'])
    # begärt pris
    df['Begärt_pris'] = pd.to_numeric(df['Begärt_pris'])
    df['Boarea']=df['Boarea'].str.replace(',','.')
    df['Boarea'] = pd.to_numeric(df['Boarea'])
    
    df['Avgift/månad'] = pd.to_numeric(df['Avgift/månad'])
    return df
    
def mdy_to_ymd(df):
    """
    Transform dateformat from hemnet to python dateformat.
    add the new date to same dataframe 
    
    df: DataFrame with date columns
    max_value: last day of the year
    min_value: fist day of the year
    """
    df['newDate']=df[str('date_sold')].str.replace(' ','')
    df['newDate']=df['newDate'].str.replace('januari','-1-')
    df['newDate']=df['newDate'].str.replace('februari','-2-')
    df['newDate']=df['newDate'].str.replace('mars','-3-')
    df['newDate']=df['newDate'].str.replace('april','-4-')
    df['newDate']=df['newDate'].str.replace('maj','-5-')
    df['newDate']=df['newDate'].str.replace('juni','-6-')
    df['newDate']=df['newDate'].str.replace('juli','-7-')
    df['newDate']=df['newDate'].str.replace('augusti','-8-')
    df['newDate']=df['newDate'].str.replace('september','-9-')
    df['newDate']=df['newDate'].str.replace('oktober','-10-')
    df['newDate']=df['newDate'].str.replace('november','-11-')
    df['newDate']=df['newDate'].str.replace('december','-12-')
    df['newDate'] = pd.to_datetime(df['newDate'])

    return df
    
    
def cyclical_transform_week(df):
    """
    Transform numerical feature to points on the unit circle.
    
    df: DataFrame with numerical columns
    date_sold: Date to transforem
    max_value: upper limit for all columns in df
    """
    df_week = df[('newDate')]
    df_week = df_week.dt.weekday
    max_value=df_week.max()
    #print(max_value)
    cyclical_df = pd.DataFrame()
    cyclical_df['sin_week'.format(df_week)] = np.sin(2 * np.pi * df_week/max_value)
    cyclical_df['cos_week'.format(df_week)] = np.cos(2 * np.pi * df_week/max_value)
    # Merge the two datasets
    modify_data = cyclical_df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df=pd.concat([df, modify_data], axis=1)
    
    return df

def cyclical_transform_day(df):
    """
    Transform numerical feature to points on the unit circle.
    
    df: DataFrame with numerical columns
    date_sold: Date to transforem
    max_value: upper limit for all columns in df
    """
    df_day = df[('newDate')]
    df_day = df_day.dt.day
    max_value=df_day.max()
    #print(max_value)
    cyclical_df = pd.DataFrame()
    cyclical_df['sin_day'.format(df_day)] = np.sin(2 * np.pi * df_day/max_value)
    cyclical_df['cos_day'.format(df_day)] = np.cos(2 * np.pi * df_day/max_value)
    
    modify_data = cyclical_df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df=pd.concat([df, modify_data], axis=1)
    
    return df


def cyclical_transform_month(date_cyclial, max_value):
    """
    Transform numerical feature to points on the unit circle.
    
    df: DataFrame with numerical columns
    max_value: upper limit for all columns in df (12)
    """
    df = date.dt.month
    cyclical_df = pd.DataFrame()
    cyclical_df['sin_month'.format(df)] = np.sin(2 * np.pi * df[df]/max_value)
    cyclical_df['cos_month'.format(df)] = np.cos(2 * np.pi * df[df]/max_value)
    #cyclical_df=pd.DataFrame(cyclical_df,columns=['sin_week', 'cos_week'])
    return cyclical_df

def uniqe_streat_name(df,colum):
    """
    Extract the streat name from streat functions
    
    df: DataFrame string values
    colum: streate name varibel 
    output: street name with out number
    """
    appended_data = []
    # Regex to extrat string untill first numeric value
    regex = r'(.*?)\d+.*?$'
    name=list(df[str(colum)])
    
    #for loop to break out streat name
    for i in name:
        res = re.findall(regex, str(i))
        #print(res)
        appended_data.append(res)
        
    # reset index on borth tabels
    df_new = pd.DataFrame(appended_data,columns=['first_streat_name'])
    df_new = df_new.reset_index(drop=True)
    df = df.reset_index(drop=True)
    
    #merge dataset 
    df=pd.concat([df, df_new['first_streat_name']], axis=1)
    return df

def encoding(df):
    """
    encode text string to numic values
    
    df: DataFrame string values
    colum: streate name varibel 
    output: street name with out number
    """
    lbl = preprocessing.LabelEncoder()
    df['encode_street'] = lbl.fit_transform(df['streetName'].fillna('').astype(str))
    df['encode_brocker_name'] = lbl.fit_transform(df['Name_brocker'].fillna('').astype(str))
    df['encode_broker'] = lbl.fit_transform(df['broker'].fillna('').astype(str))
    df['encode_area'] = lbl.fit_transform(df['area'].fillna('').astype(str))
    df['encode_first_streat_name'] = lbl.fit_transform(df['first_streat_name'].fillna('').astype(str))
    return df