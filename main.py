#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
import tensorflow as tf
import os
import time
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print("Numpy version = ", np.__version__)
print("Tensorflow version = ", tf.__version__)
print("Sklearn version = ", skl.__version__)
print("Pandas version = ", pd.__version__)
print("Python version = ", __import__("platform").python_version())

warnings.filterwarnings('ignore')

def print_frequency(y_train,y_test,kind='Validation'):
    unique, counts = np.unique(y_train, return_counts=True)
    uniquet, countst = np.unique(y_test, return_counts=True)

    fig, ax = plt.subplots()
    rects1 = ax.bar(unique - 0.2, counts, 0.25, label='Treinamento')
    rects2 = ax.bar(unique + 0.2, countst, 0.25, label=kind)
    ax.legend()
    ax.set_xticks(unique)
    ax.set_xticklabels(labels)

    plt.title('Dataset Fashion MNIST')
    plt.xlabel('Clase')
    plt.ylabel('Frequência')
    plt.savefig('frequency.png')
    plt.show()

def create_confusion_matrix (y_val, y_predict, score, vmax, model):
    cm = skl.metrics.confusion_matrix(y_val, y_predict)

    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square = True, cmap = 'Blues_r',vmin=0,vmax=vmax);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = model+'\n\nAccuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15);

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

sprite = {
        0: 'T-shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

labels = ["%s" % i for i in range(10)]

# print_frequency(y_train,y_test,kind='Teste')

fig, ax = plt.subplots(2, 4, figsize = (12, 6))

for i in range(8):
    ax[i//4, i%4].imshow(x_train[i], cmap='gray')
    ax[i//4, i%4].axis('off')
    ax[i//4, i%4].set_title("Class %d: %s" 
                            %(y_train[i],sprite[y_train[i]]))
    
# plt.show()


x_train = (x_train/255.0).astype('float32').reshape((60000,28*28))
x_test = (x_test/255.0).astype('float32').reshape((10000,28*28))


N, d = x_train.shape
index = np.arange(N)

x_Dtrain,D_val,y_Dtrain,y_Dval,index_Dtrain,index_Dval = train_test_split(x_train,y_train,index,train_size=0.80,random_state=4,stratify=y_train)

# print_frequency(y_Dtrain,y_Dval)

# print("Shape of X_train:  ", x_train.shape)
# print("Shape of y_train:  ", y_train.shape)
# print("Shape of x_Dtrain:  ", x_Dtrain.shape)
# print("Shape of y_Dtrain: ", y_Dtrain.shape)
# print("Shape of D_val:    ", D_val.shape)
# print("Shape of y_Dval:   ", y_Dval.shape)

u, c_y_train = np.unique(y_train, return_counts=True)
u, c_y_dtrain = np.unique(y_Dtrain, return_counts=True)
# print("Proportion of classes in y_train:  ", c_y_train[0:10]/len(y_train))
# print()
# print("Proportion of classes in y_Dtrain: ", c_y_dtrain[0:10]/len(y_Dtrain))
# print()
# print("Differences in proportions of classes in y_train and y_Dtrain: ", 
#       np.abs(c_y_train[0:10]/len(y_train) - c_y_dtrain[0:10]/len(y_Dtrain)))
# print()
# print("Sum of proportions in y_train = ", sum(c_y_train[0:10]/len(y_train)))



# #Logistic Regression

# model_lgreg = LogisticRegression()
# solvers_lgreg = ['newton-cg','lbfgs']
# c_values_lgreg = [100.0, 50.0, 10.0, 5.0, 1.0, 0.1, 0.01] 
# grid_lgreg = dict(solver=solvers_lgreg,C=c_values_lgreg,random_state=[4])

# time_lgreg = time.time()
# grid_search_lgreg = GridSearchCV(estimator=model_lgreg, param_grid=grid_lgreg, scoring='accuracy',verbose=3, 
#             cv=skl.model_selection.StratifiedKFold(n_splits=4,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
# grid_result_lgreg = grid_search_lgreg.fit(x_Dtrain, y_Dtrain) 

# time_lgreg = time.time() - time_lgreg

# pd.DataFrame(grid_result_lgreg.cv_results_)[['params','rank_test_score','mean_test_score']].sort_values(by=['rank_test_score'])

#SVM

model_SVM = SVC()
kernel_SVM = ['linear','rbf']
c_values_SVM = [100.0, 50.0, 10.0, 5.0, 1.0, 0.1, 0.01] 
grid_SVM = dict(kernel=kernel_SVM,C=c_values_SVM,random_state=[4])

time_SVM = time.time()
grid_search_SVM = GridSearchCV(estimator=model_SVM, param_grid=grid_SVM, scoring='accuracy',verbose=3, 
            cv=skl.model_selection.StratifiedKFold(n_splits=4,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
grid_result_SVM = grid_search_SVM.fit(x_Dtrain, y_Dtrain) 

time_SVM = time.time() - time_SVM


pd.DataFrame(grid_result_SVM.cv_results_)[['params','rank_test_score','mean_test_score']].sort_values(by=['rank_test_score'])





