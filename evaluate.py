"""
Created on Sat Dec 18 12:53:31 2021

@author: abdulatif albaseer
"""

import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn import metrics

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def classifer_preformance(clf, X_test, Y_test, y_pred):
  # #Import scikit-learn metrics module for accuracy calculation
  # from sklearn import metrics
  # from sklearn.metrics import plot_confusion_matrix
  # Model Accuracy: how often is the classifier correct?
  Acc = metrics.accuracy_score(Y_test, y_pred)
  print("Accuracy:",Acc)

  # Model Precision: what percentage of positive tuples are labeled as such?
  Pre = metrics.precision_score(Y_test, y_pred,average='macro')
  print("Precision:",Pre)

  # Model Recall: what percentage of positive tuples are labelled as such?
  Rec = metrics.recall_score(Y_test, y_pred,average='macro')
  print("Recall:",Rec)

  # Model Fscore: 
  F1 = metrics.f1_score(Y_test, y_pred,average='macro')
  print("F1 score:",F1)
  #f1_score(y_true, y_pred, average=None)
  from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
  #print(confusion_matrix(y_test,y_pred))
  print(classification_report(Y_test,y_pred))
  #print(accuracy_score(Y_test, y_pred))

  #plot_confusion_matrix(clf, X_test, Y_test)  
  cm = confusion_matrix(Y_test, y_pred)
  plot_confusion_matrix(cm, np.unique(Y_test))

  plt.show()

  return Acc, Pre, Rec, F1