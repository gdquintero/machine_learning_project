#!/usr/bin/env python
# coding: utf-8

from sqlite3 import Time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as skl
import tensorflow as tf
import os
import time
import pathlib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print("------------------")
print("Inicio do programa")
print("------------------")
print("Versao do Python: ", __import__("platform").python_version())
print("Versao da biblioteca Numpy: ", np.__version__)
print("Versao da biblioteca Matplotlib: ", np.__version__)
print("Versao da biblioteca Pandas: ", pd.__version__)
print("Versao da biblioteca Seaborn: ", np.__version__)
print("Versao da biblioteca Tensorflow: ", tf.__version__)
print("Versao da biblioteca Scikit-learn: ", skl.__version__)

warnings.filterwarnings('ignore')

# Caminho do main.py
localPath = pathlib.Path(__file__).parent.resolve()

# Função para plotar a frequência com que cada uma das classes aparece
def getFrequency(y_train,y_test,kind=None):
    classes, countTrain = np.unique(y_train, return_counts=True)
    countTest = np.unique(y_test, return_counts=True)[1]
    width = 0.3 # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(classes - width/2, countTrain, width, label='Train')
    ax.bar(classes + width/2, countTest, width, label=kind)
    plt.title('Fashion MNIST')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    ax.set_xticks(classes)
    ax.legend()
    fig.tight_layout()

    if kind == 'Test':
        plt.savefig(str(localPath) + '/frequency.png')
    else:
        plt.savefig(str(localPath) + '/frequencyCV.png')

def getImages(x_img,y_img):
    ax = plt.subplots(2, 4, figsize = (12, 6))[1]

    for i in range(8):
        ax[i//4, i%4].imshow(x_img[i], cmap='gray')
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title("Class %d: %s" %(y_img[i],sprite[y_img[i]]))
    plt.savefig(str(localPath) + '/sprite.png')
    # plt.show()

# Funcao para montar a matriz de confusao
def getConfusionMatrix (y_val, y_predict, score, vmax, model):
    cm = skl.metrics.confusion_matrix(y_val, y_predict)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square = True, cmap = 'Blues_r',vmin=0,vmax=vmax);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = model+'\n\nAccuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15);

# Carregamento dos dados
print("\n-----------------------------------")
print("Carregando o dataset: Fashion mnist")
print("-----------------------------------")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("Numero de amostras para treinamento: ",x_train.shape[0])
print("Numero de amostras para teste: ",x_test.shape[0])
print("Tamanha de cada amostra: ",x_train[0,:,:].shape," pixels de escala de cinza")

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

getFrequency(y_train,y_test,kind='Test')
getImages(x_train,y_train)

# Normalizacao
x_train = (x_train/255.0).astype('float32').reshape((60000,28*28))
x_test = (x_test/255.0).astype('float32').reshape((10000,28*28))

N, d = x_train.shape
index = np.arange(N)

# Dividindo o conjunto de treinamento para a CV
print("\n---------------------------------------------")
print("Dividindo o conjunto de treinamento para a CV")
print("---------------------------------------------")
x_Dtrain,D_val,y_Dtrain,y_Dval,index_Dtrain,index_Dval = train_test_split(x_train,y_train,index,train_size=0.80,random_state=4,stratify=y_train)

print("Numero de amostras para treinamento: ",x_Dtrain.shape[0])
print("Numero de amostras para validacao: ",D_val.shape[0])

getFrequency(y_Dtrain,y_Dval,kind='Validation')

#Logistic Regression
print("\n-------------------------------------------")
print("Ajuste do modelo usando Regressao Logistica")
print("-------------------------------------------")
model_lgreg = LogisticRegression()
solvers_lgreg = ['newton-cg','lbfgs']
c_values_lgreg = [0.01] 
grid_lgreg = dict(solver=solvers_lgreg,C=c_values_lgreg,random_state=[4])

time_lgreg = time.time()
grid_search_lgreg = GridSearchCV(estimator=model_lgreg, param_grid=grid_lgreg, scoring='accuracy',verbose=3, 
            cv=skl.model_selection.StratifiedKFold(n_splits=2,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
grid_result_lgreg = grid_search_lgreg.fit(x_Dtrain, y_Dtrain) 

time_lgreg = time.time() - time_lgreg

pd.DataFrame(grid_result_lgreg.cv_results_)[['params','rank_test_score','mean_test_score']].sort_values(by=['rank_test_score'])

print("\nTempo de execucao: ",int(time_lgreg),"segundos ou",round(time_lgreg/60,2),"minutos")







