"""
Created on Sat Dec 18 12:53:31 2021

@author: abdulatif albaseer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from os import chdir as cd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pathlib2 as pl2
from Attack_Funcs import *
from data_util import *
from data_processing import *
from evaluate import *
from visual_data import *

head=['Delivery Date',	'Generator',	'Fuel Type',	'Measurement',	'Hour 1',	'Hour 2',	'Hour 3',	'Hour 4',	'Hour 5',	'Hour 6',	'Hour 7',	'Hour 8',	'Hour 9',	'Hour 10',	'Hour 11',	'Hour 12',	'Hour 13',	'Hour 14',	'Hour 15',	'Hour 16',	'Hour 17',	'Hour 18',	'Hour 19',	'Hour 20',	'Hour 21','Hour 22',	'Hour 23',	'Hour 24']

year19 = df_concat('./Data New/2019')

year20 = df_concat('./Data New/2020')

year21 = df_concat('./Data New/2021')


"Combine Data "
df = pd.concat([year19,year20,year21])
to_attack = pd.concat([year19,year20,year21])         #pd.concat([year19])


df = df_pre_processing(df)# Benign 
to_attack = df_pre_processing(to_attack)# Attack


# # Data Processing 

df = df_processing(df)

to_attack = df_processing(to_attack)

df = df_outlier(df)

to_attack = df_outlier(to_attack)


df.to_csv('benign_ready.csv',index=False)
to_attack.to_csv('attacked_ready.csv',index=False)
