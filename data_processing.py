
"""
Created on Sat Dec 18 12:53:31 2021

@author: abdulatif albaseer
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
def df_pre_processing(df):
    
  df["Delivery Date"]=pd.to_datetime(df["Delivery Date"])
  df["Dayofweek"]=pd.DatetimeIndex(df["Delivery Date"]).dayofweek
  df["month"]=pd.DatetimeIndex(df["Delivery Date"]).month
  "replace empty with NAN"
  df = df.replace(r"^\s*$", np.nan, regex=True)
  "Select Output measurment"
  df = (df[df['Measurement'] == 'Output' ] )
  df.reset_index(inplace=True)
  "Select SOLAR Fuel Type"
  df = (df[df['Fuel Type'] == 'SOLAR' ] )
  df.reset_index(inplace=True)
  "Drop col"
  df.drop(columns=['Measurement','level_0','Delivery Date','Fuel Type','index','Generator' ],inplace=True)
  "Convert Object data to numeirc"
  df = df.apply(pd.to_numeric)
  return df


def df_processing(df):
  "Drop tuples with zeros all day record"
  df = df.loc[~(df==0).iloc[:,0:-2].all(axis=1)]
  print("Data shape after Drop tuples with zeros all day record:")
  print(df.shape)
  "Drop tuples with NaN all day record"
  df = df.dropna(how='all',subset=['Hour 1',	'Hour 2',	'Hour 3',	'Hour 4',	'Hour 5',	'Hour 6',	'Hour 7',	'Hour 8',	'Hour 9',	'Hour 10',	'Hour 11',	'Hour 12',	'Hour 13',	'Hour 14',	'Hour 15',	'Hour 16',	'Hour 17',	'Hour 18',	'Hour 19',	'Hour 20',	'Hour 21','Hour 22',	'Hour 23',	'Hour 24'])
  print("Data shape after Drop tuples with NaN all day record:")
  print(df.shape)
  "Fill Nan with the median of the day "
  print('Check if we have any null value, number of null in each col')
  print(df.isnull().sum())
  print('Fill null with median of each day...')
  df.fillna(df.median(),inplace=True)
  print('Check if we have any null value, number of null in each col')
  print(df.isnull().sum())

  return df  

def df_outlier(df):
  " Dealing with outlier IQR method"
  #'Hour 1',	'Hour 2',	'Hour 3',	'Hour 4',	'Hour 5',	'Hour 6',	'Hour 7', ,	'Hour 19',	'Hour 20',	'Hour 21','Hour 22',	'Hour 23',	'Hour 24'
  col_name=['Hour 1',	'Hour 2',	'Hour 3',	'Hour 4',	'Hour 5',	'Hour 6',	'Hour 7','Hour 8',	'Hour 9',	'Hour 10',	'Hour 11',	'Hour 12',	'Hour 13',	'Hour 14',	'Hour 15',	'Hour 16',	'Hour 17',	'Hour 18',	'Hour 19',	'Hour 20',	'Hour 21','Hour 22',	'Hour 23',	'Hour 24']
  #
  print("Data shape before outlier removing")
  print(df.shape)
  print('removing outliers')
  for i in tqdm(range(len(col_name))):
    df = imput_mean_outlier(df, col_name[i])
  print('Done')
  print("Data shape after outlier removing")
  print(df.shape)

  return df


def imput_mean_outlier(df_in, col_name):
  for i in df_in[col_name]:
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    if i > fence_high or i < fence_low:
      df_in[col_name] = df_in[col_name].replace(i, np.mean(df_in[col_name]))
      
  return df_in