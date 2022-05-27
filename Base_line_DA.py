
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
import pandas as pd
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn import svm,tree
# # Read Benign and Attacked datasets 

df = pd.read_csv('benign_ready.csv')

to_attack = pd.read_csv('attacked_ready.csv')


B = df.to_numpy()



"Attack generation"
Attack = to_attack.to_numpy()

alpha, l, h = 0.2, 0.15, 0.25

aA = Attack_generation(Attack[:,0:-2], alpha, l, h)# exclude month information from attack 


mA=np.concatenate([Attack[:,-2:],Attack[:,-2:],Attack[:,-2:],Attack[:,-2:]],axis=0)
mA.shape

print(Attack[:,-2:])
B.shape


"X and Y "
A=np.concatenate([aA,mA],axis=1)
#===================================
X = np.concatenate((B,A))

Y = np.concatenate((np.zeros(len(B)),np.ones(len(A))))



print('Shape of Feature Matrix:', X.shape)
print('Shape of Target Vector:', Y.shape)

print('Original Target Variable Distribution:', Counter(Y))


ada = ADASYN(sampling_strategy='minority', random_state= 420 , n_neighbors = 5)
par=ada.get_params(deep=True)
X_res, Y_res = ada.fit_resample(X,Y)

print('Oversampled Target Variable Distribution:', Counter(Y_res))


"Data spliting "

#last 3 month as test data 
X = X_res.copy()#sk.preprocessing.normalize(X_res, norm="l1")
Y = Y_res.copy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train.shape

" feature standrization "

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# clf = svm.SVC(kernel='linear') # Linear Kernel
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)
# Apply attack on the benign data
X_test_t=X_test.copy()
Y_test_t=Y_test.copy()
classifer_preformance(clf, X_test, Y_test, y_pred)

count = 0
for i in range(len(Y_test_t)//2):
    if Y_test_t[i]==0.0:
        X_test_t[i]=X_test_t[i]+X_test_t[i]*0.3
        Y_test_t[i] = 1.0
        count+=1      

print('count',count)

# add submission file 
print("alpha, l, h")
print(alpha,l,h)
print(" ")
print("hyper paramter")

print("")
print("Network preformance")
classifer_preformance(clf, X_test, Y_test, y_pred)

