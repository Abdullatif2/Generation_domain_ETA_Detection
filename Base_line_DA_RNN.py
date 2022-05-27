# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 10:57:22 2021

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


B.shape


"X and Y "
A=np.concatenate([mA,aA],axis=1)
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


# # Deep learning: RNN, GRU, LSTM


"Get Data in good shape for RNN: tensor[X][Y]"
#=========Train Dataset====================
# our data in list format we need to convert to tensor
my_xtrain=[]
my_ytrain=[]
# step 1: convert to numpy array
for i in range(len(X_train)):
  #print(i)
  my_xtrain.append(np.array(X_train[i]))
  my_ytrain.append(Y_train[i])
# step 2: convert to tensor
tensor_xtrain = torch.Tensor(my_xtrain)
tensor_ytrain = torch.Tensor(my_ytrain).type(torch.LongTensor)# make sure the y is 1 D
#=========Test Dataset====================
my_xtest=[]
my_ytest=[]
for i in range(len(X_test)):
  #print(i)
  my_xtest.append(np.array(X_test[i]))
  my_ytest.append(Y_test[i])

tensor_xtest = torch.Tensor(my_xtest)
tensor_ytest = torch.Tensor(my_ytest).type(torch.LongTensor)


"Creat data batchs"
bs=128
from torch.utils.data import TensorDataset, DataLoader
train_dataset1 = TensorDataset(tensor_xtrain,tensor_ytrain)
Train_loader= DataLoader(train_dataset1, batch_size=bs, shuffle=True)

test_dataset1 = TensorDataset(tensor_xtest,tensor_ytest)
Test_loader= DataLoader(test_dataset1, batch_size=bs, shuffle=False)

"Hyper parameter: check dr link"
# Device configuration
# if you have gpu support 
# we will push our tensor to the device to make sure it is running on gpu
# the gpu is good for parallel calculation (should we use cpu with RNN)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
hidden_size = 128 # you can try diffrent sizes here 
num_classes = 2 #10
num_epochs = 10
batch_size = bs #100 # make sure this is < your entier data size
learning_rate = 0.0001 # 0.001

input_size = 1 #28 # we tack each row of the image 
sequence_length = 24 #28 # number of features 
num_layers = 4

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #======= Change model ============
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)#,bidirectional=True
        #self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #=================================
        # the shape we need to our input  x -> (batch_size, sequence, input_size)
        # adding the classification step 
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # RNN input: x and h0 in the correct shape
        # no activation and no softmax at the end 
        # becuase we apply crossentropy loss function will do the softmax
        h0 = torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)
        #h0 = torch.cat((h0[-2,:,:], h0[-1,:,:]), dim = 1) 
        #c0 = torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(device)
        out, _ = self.rnn(x,h0)
        #out, _ = self.rnn(x,(h0,c0))
        # batch size, sequ, hidden size
        # out(N, 28, 128)
        # only need the last time step for calssification 
        out = out[:,-1,:]
        out = self.fc(out)# we do the classification depending on the last output of the hidden layer 

        return out

model = RNN(input_size, hidden_size,num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()# sigmoid 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(Train_loader)
for epoch in range(num_epochs):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
    for i, (images,labels) in enumerate(Train_loader):
        images = images.reshape(-1, sequence_length,input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        n_samples = labels.size(0)
        n_correct = (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        acc = 100.0 * n_correct / n_samples
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], accu: [{acc}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    all_preds = torch.tensor([])
    n_correct = 0
    n_samples = 0
    for images, labels in Test_loader:
        images = images.reshape(-1, sequence_length,input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        all_preds = torch.cat(
            (all_preds, predicted)
            ,dim=0
        )
       # y_pred.append(y_pred,predicted)

    acc = 100.0 * n_correct / n_samples

    print(f'Accuracy of the network: {acc} %')
    print(f'n_correct: {n_correct} ')
    print(f'n_samples: {n_samples} ')


# In[ ]:


# add submission file 
print("alpha, l, h")
print(alpha,l,h)
print(" ")
print("hyper paramter")
print("batchsize",bs)
print("epochs",num_epochs)
print("layers",num_layers)
print("learning rate",learning_rate)
print("")
print("Network preformance")
classifer_preformance(model, X_test, Y_test, all_preds)


# In[47]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y_true = Y_test
y_pred = all_preds

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k')
plt.plot(fpr, tpr, label="GRU RNN(area = {:.3f})".format(roc_auc))
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


"Attack generation"
alphat, lt, ht = 0.08,0.03, 0.13
# ceraet realistic data attack<<< benign
to_attack_test=df.sample(n=300).to_numpy()
aAt = Data_generation(to_attack_test[:,0:-2], alphat, lt, ht)# exclude month information from attack 

mAt=np.concatenate([to_attack_test[:,-2:],to_attack_test[:,-2:],to_attack_test[:,-2:],to_attack_test[:,-2:]],axis=0)
mAt.shape

"X and Y "
At=np.concatenate([mAt,aAt],axis=1)
#===================================
Xt = np.concatenate((B,At))
#np.shape(X)#(216, 24)
Yt = np.concatenate((np.zeros(len(B)),np.ones(len(At))))
#np.shape(Y)#(216,)

"Data spliting "
from sklearn.model_selection import train_test_split
import sklearn as sk

#X = sk.preprocessing.normalize(X_res, norm="l1")
#Y = Y_res
X_traint, X_testt, Y_traint, Y_testt = train_test_split(Xt, Yt,test_size=0.5, random_state=42)


" feature standrization "
# case data are from same source
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()#StandardScaler()
X_traint = sc.fit_transform(X_traint)
X_testt = sc.transform(X_testt)

"Get Data in good shape for RNN: tensor[X][Y]"
#=========Train Dataset====================
# our data in list format we need to convert to tensor
my_xtraint=[]
my_ytraint=[]
# step 1: convert to numpy array
for i in range(len(X_traint)):
  #print(i)
  my_xtraint.append(np.array(X_traint[i]))
  my_ytraint.append(Y_traint[i])
# step 2: convert to tensor
tensor_xtraint = torch.Tensor(my_xtraint)
tensor_ytraint = torch.Tensor(my_ytraint).type(torch.LongTensor)# make sure the y is 1 D
#=========Test Dataset====================
my_xtestt=[]
my_ytestt=[]
for i in range(len(X_testt)):
  #print(i)
  my_xtestt.append(np.array(X_testt[i]))
  my_ytestt.append(Y_testt[i])

tensor_xtestt = torch.Tensor(my_xtestt)
tensor_ytestt = torch.Tensor(my_ytestt).type(torch.LongTensor)


"Creat data batchs"

from torch.utils.data import TensorDataset, DataLoader
train_dataset1t = TensorDataset(tensor_xtraint,tensor_ytraint)
Train_loadert= DataLoader(train_dataset1t, batch_size=bs, shuffle=True)

test_dataset1t = TensorDataset(tensor_xtestt,tensor_ytestt)
Test_loadert= DataLoader(test_dataset1t, batch_size=bs, shuffle=False)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
import torch.nn.functional as F
with torch.no_grad():
    all_predst = torch.tensor([])
    all_prop = torch.tensor([])
    n_correct = 0
    n_samples = 0
    for images, labels in Test_loadert:
        images = images.reshape(-1, sequence_length,input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        prop_p = F.softmax(outputs, dim=1)
        all_prop = torch.cat(
            (all_prop,prop_p), dim=0
        )
        prop_p, top_class = prop_p.topk(1,dim=1)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        all_predst = torch.cat(
            (all_predst, predicted)
            ,dim=0
        )

    acc = 100.0 * n_correct / n_samples

    print(f'Accuracy of the network: {acc} %')
    print(f'n_correct: {n_correct} ')
    print(f'n_samples: {n_samples} ')


# ## Preformance...

TestSum=pd.DataFrame()
TestSum["Prob"]=all_prop
TestSum["Prediction"]=all_predst
TestSum["True class"]=Y_testt
TestSum.to_csv('/content/gdrive/MyDrive/master/pro008_003_013.csv')

print("testing with realistic data")
print("alpha, l, h")
print(alphat,lt,ht)
print(" ")
print("hyper paramter")
print("batchsize",bs)
print("epochs",num_epochs)
print("layers",num_layers)
print("learning rate",learning_rate)
print("")
print("Network preformance")
classifer_preformance(model, X_testt, Y_testt, all_predst)

alphab, lb, hb = 0.02, 0.01, 0.07
###############################################################################
aAb = Data_generation(Attack[:,0:-2], alphab, lb, hb)# exclude month information from attack 
mAb=np.concatenate([Attack[:,-2:],Attack[:,-2:],Attack[:,-2:],Attack[:,-2:]],axis=0)
mAb.shape
#mA=np.reshape(mA, (-1,1))
"X and Y "
Ab=np.concatenate([mAb,aAb],axis=1)
#===================================
Xb = np.concatenate((B,Ab))
#np.shape(X)#(216, 24)
Yb = np.concatenate((np.zeros(len(B)),np.ones(len(Ab))))
#np.shape(Y)#(216,)
"Data spliting "
from sklearn.model_selection import train_test_split
import sklearn as sk
#last 3 month as test data 
#X = sk.preprocessing.normalize(X_res, norm="l1")
#Y = Y_res
X_trainb, X_testb, Y_trainb, Y_testb = train_test_split(Xb, Yb, test_size=0.1, random_state=42)
" feature standrization "
# case data are from same source
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()#StandardScaler()
X_trainb = sc.fit_transform(X_trainb)
X_testb = sc.transform(X_testb)

# case data from diffrent source, do subject wise normalization 

"Get Data in good shape for RNN: tensor[X][Y]"
#=========Train Dataset====================
# our data in list format we need to convert to tensor
my_xtrainb=[]
my_ytrainb=[]
# step 1: convert to numpy array
for i in range(len(X_trainb)):
  #print(i)
  my_xtrainb.append(np.array(X_trainb[i]))
  my_ytrainb.append(Y_trainb[i])
# step 2: convert to tensor
tensor_xtrainb = torch.Tensor(my_xtrainb)
tensor_ytrainb = torch.Tensor(my_ytrainb).type(torch.LongTensor)# make sure the y is 1 D
#=========Test Dataset====================
my_xtestb=[]
my_ytestb=[]
for i in range(len(X_testb)):
  #print(i)
  my_xtestb.append(np.array(X_testb[i]))
  my_ytestb.append(Y_testb[i])

tensor_xtestb = torch.Tensor(my_xtestb)
tensor_ytestb = torch.Tensor(my_ytestb).type(torch.LongTensor)
"Creat data batchs"

from torch.utils.data import TensorDataset, DataLoader
train_dataset1b = TensorDataset(tensor_xtrainb,tensor_ytrainb)
Train_loaderb= DataLoader(train_dataset1b, batch_size=bs, shuffle=True)

test_dataset1b = TensorDataset(tensor_xtest,tensor_ytest)
Test_loaderb= DataLoader(test_dataset1b, batch_size=bs, shuffle=False)




# In[ ]:


#Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    all_predsb = torch.tensor([])
    n_correct = 0
    n_samples = 0
    for images, labels in Test_loaderb:
        images = images.reshape(-1, sequence_length,input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        all_predsb = torch.cat(
            (all_predsb, predicted)
            ,dim=0
        )

    acc = 100.0 * n_correct / n_samples

    print(f'Accuracy of the network: {acc} %')
    print(f'n_correct: {n_correct} ')
    print(f'n_samples: {n_samples} ')


# In[ ]:


print("testing with Balanced data")
print("alpha, l, h")
print(alphab,lb,hb)
print(" ")
print("hyper paramter")
print("batchsize",bs)
print("epochs",num_epochs)
print("layers",num_layers)
print("learning rate",learning_rate)
print("")
print("Network preformance")
classifer_preformance(model, X_testb, Y_testb, all_predsb)


# In[ ]:


all_preds.shape

