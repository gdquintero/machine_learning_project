#!/usr/bin/env python
# coding: utf-8

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
from sklearn.neural_network import MLPClassifier

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

def export(yVal,yPred,dataFrame,n):
    with open('latex.txt','w') as f:
        for i in range(n):
            C = dataFrame.at[i,'params']['C']
            solver = dataFrame.at[i,'params']['solver']
            mScore = dataFrame.at[i,'mean_test_score']
            f.write("%i %s %3.2f %s %s %s %1.4f %s" % (i+1,'&',C,'&',solver,'&',mScore,'\\\\\n'))
    f.close()

    cm = skl.metrics.confusion_matrix(yVal, yPred)

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

# Funcao para printar as primeiras 8 imagens do dataset
def getImages(x_img,y_img):
    ax = plt.subplots(2, 4, figsize = (12, 6))[1]

    for i in range(8):
        ax[i//4, i%4].imshow(x_img[i], cmap='gray')
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title("Class %d: %s" %(y_img[i],sprite[y_img[i]]))
    plt.savefig(str(localPath) + '/sprite.png')

# Funcao para montar a matriz de confusao
def getConfusionMatrix (y_val, y_predict, score, vmax, model):
    cm = skl.metrics.confusion_matrix(y_val, y_predict)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square = True, cmap = 'Blues_r',vmin=0,vmax=vmax)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = model+'\n\nAccuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig(str(localPath) + '/confusionmatrix.png')

def test1(trainSize):
        # Dividindo o conjunto de treinamento para a CV
    print("\nDividindo o conjunto de treinamento para a CV")
    print("---------------------------------------------")
    x_Dtrain,D_val,y_Dtrain,y_Dval,index_Dtrain,index_Dval = train_test_split(x_train,y_train,index,train_size=trainSize,random_state=4,stratify=y_train)

    print("Numero de amostras para treinamento: ",x_Dtrain.shape[0])
    print("Numero de amostras para validacao: ",D_val.shape[0])

    getFrequency(y_Dtrain,y_Dval,kind='Validation')

    #Logistic Regression
    print("\nAjuste do modelo usando Regressao Logistica")
    print("-------------------------------------------")
    modelLR = LogisticRegression()
    solversLR = ['lbfgs']
    rowTable = len(solversLR)
    cLR = [0.01] 
    rowTable = rowTable * len(cLR)
    gridLR = dict(solver=solversLR,C=cLR,random_state=[4])

    timeLR = time.time()

    gridSearchLR = GridSearchCV(estimator=modelLR, param_grid=gridLR, scoring='accuracy',verbose=3, 
                cv=skl.model_selection.StratifiedKFold(n_splits=2,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
    gridResultLR = gridSearchLR.fit(x_Dtrain, y_Dtrain) 
    df = pd.DataFrame(gridResultLR.cv_results_)[['params','rank_test_score','mean_test_score']].sort_values(by=['rank_test_score'])

    timeLR = time.time() - timeLR

    print("\nTempo de execucao RL: ",int(timeLR),"segundos ou",round(timeLR/60,2),"minutos")

    #SVM
    # print("\nAjuste do modelo usando SVM")
    # print("--------------------------------")
    # modelSVM = SVC()
    # kernelSVM = ['sigmoid']
    # cSVM = [100.0] 
    # gridSVM = dict(kernel=kernelSVM,C=cSVM,random_state=[4])

    # timeSVM = time.time()
    # gridSearchSVM = GridSearchCV(estimator=modelSVM, param_grid=gridSVM, scoring='accuracy',verbose=3, 
    #             cv=skl.model_selection.StratifiedKFold(n_splits=2,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
    # gridResultSVM = gridSearchSVM.fit(x_Dtrain, y_Dtrain) 

    # timeSVM = time.time() - timeSVM

    # print("\nTempo de execucao SVM: ",int(timeSVM),"segundos ou",round(timeSVM/60,2),"minutos")

    # print("\nTempo total de execucao: ",timeLR + timeSVM)

    yDvalPredLR = gridResultLR.predict(D_val)
    # y_Dval_predict_SVM   = gridResultSVM.predict(D_val)

    scoreDvalLR = gridResultLR.score(D_val,y_Dval)
    # score_Dval_SVM   = gridResultSVM.score(D_val,y_Dval)

    export(y_Dval,yDvalPredLR,df,rowTable)

    # # getConfusionMatrix(y_Dval, y_Dval_predict_lgreg, score_Dval_lgreg, 2000,'Logistic Regression')
    # # getConfusionMatrix(y_Dval, y_Dval_predict_SVM, score_Dval_SVM, 2000, 'SVM')

    # if score_Dval_lgreg > score_Dval_SVM:
    #     final_model = gridResultLR
    # else:
    #     final_model = gridResultSVM

    # # score_Dval_final_model = score_Dval_lgreg  if score_Dval_lgreg > score_Dval_SVM else score_Dval_SVM

    # final_model = final_model.best_estimator_
    # print("O melhor modelo e: ", final_model)
    

# Carregamento dos dados
print("\nCarregando o dataset: Fashion mnist")
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

print("\n-------------------")
print("Execucao do teste 1")
print("-------------------")
test1(0.80)
