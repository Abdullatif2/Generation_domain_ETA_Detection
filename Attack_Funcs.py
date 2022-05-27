
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:53:31 2021

@author: abdulatif albaseer
"""
import numpy as np
import pandas as pd
"Dataset generation"
def Attack_generation(wpp, alpha, l, h):
  "Benige Data "
  # add labels and save in .csv file 
  # label_Benige = np.zeros((30,1))
  # Benige = np.concatenate((wpp, label_Benige), axis=1)
  df_wpp = pd.DataFrame(wpp)

  "Type 1 Attack: Partial Increment Attack (fixed alpha)"
  df_1 = wpp + wpp*alpha # function 
  #df_1 = np.concatenate((df_1, label_Attack1), axis=1)

  "Type 2 Attack: Partial Increment Attack (random alpha)"
  # l=0.2 #0.15
  # h=0.3 # 0.35
  #label_Attack2 = 2*np.ones((30,1))
  N=[len(wpp),np.size(wpp, axis=1)]
  alpha_rand = np.ones(N)
  np.random.seed(100)
  alpha_rand *= np.random.uniform(low=l, high=h, size=N)
  df_2 = wpp + wpp*alpha_rand # function 
  #df_2 = np.concatenate((df_2, label_Attack1), axis=1)

  "Type 3 Attack: Minimum Generation Attack"
  #label_Attack3 = 3*np.ones((30,1))
  #alpha = 0.2 
  df_3=df_wpp
  # replace min(row) with 0 to get exact same attack in LR
  df_3.apply(lambda row: row.replace(min(row), max(row)*alpha), axis=1)
  #df_3 = pd.concat([df_3,pd.DataFrame(label_Attack1)], axis = 1)
  df_3 = df_3.to_numpy()

  "Type 4 Attack: Peak Generation Attack"
  #label_Attack4 = 4*np.ones((30,1))
  df_4= wpp
  for i in range(len(wpp)):
    row = wpp[i,:] 
    loca = np.argmax(row)
    df_4[i,loca:] = max(row)
  #df_4 = np.concatenate((df_4, label_Attack1), axis=1)

  "Collect all in one"
  # np concatnate function work better than pd.concat 
  A = np.concatenate([df_1,df_2,df_3,df_4], axis=0)
  return A
  # returen np matrix