# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:34:28 2022

@author: amalb
"""
###### Reach 81 % with 10 epochs
from Attack_Funcs import *
from data_util import *
from data_processing import *
from evaluate import *
from visual_data import *
from sklearn.model_selection import train_test_split
import sklearn as sk
import torch
from nn_models import RNN
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from collections import Counter

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

X = X_res[:,:-2]#sk.preprocessing.normalize(X_res, norm="l1")
Y = Y_res
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# X_train=X_train[:,:-2]
# X_test=X_test[:,:-2]

sc = MinMaxScaler() #StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


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
for i in tqdm(range(len(X_test))):
  #print(i)
  my_xtest.append(np.array(X_test[i]))
  my_ytest.append(Y_test[i])

tensor_xtest = torch.Tensor(my_xtest)
tensor_ytest = torch.Tensor(my_ytest).type(torch.LongTensor)


"Creat data batchs"
bs=16

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



model = RNN(input_size, hidden_size,num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()# sigmoid 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(Train_loader)
for epoch in tqdm(range(num_epochs)):
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
    all_preds = torch.tensor([]).to(device)
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