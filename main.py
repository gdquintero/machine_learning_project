#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
import tensorflow as tf
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
print("Versao da biblioteca Tensorflow: ", tf.__version__)
print("Versao da biblioteca Scikit-learn: ", skl.__version__)

warnings.filterwarnings('ignore')

# Caminho do main.py
localPath = pathlib.Path(__file__).parent.resolve()

def export(test,yVal,yPredLR,yPredSVM,yTest,testPredFinal,testPredFinalTotal,dataFrameLR,dataFrameSVM,nLR,nSVM,outparam):
    with open('test'+str(test),'w') as f:
        f.write("%s %i\n" % ('Test ',test))
        f.write("\n")
        f.write("Ordenacao dos scores Regressao Logistica:\n")
        for i in range(nLR):
            C = dataFrameLR.at[i,'params']['C']
            solver = dataFrameLR.at[i,'params']['solver']
            mScore = dataFrameLR.at[i,'mean_test_score']
            f.write("%i %s %3.2f %s %s %s %1.4f %s" % (i+1,'&',C,'&',solver,'&',mScore,'\\\\\n'))
        f.write("\n")
        f.write("Matriz de custo Regressao Logistica:\n")
        cm = skl.metrics.confusion_matrix(yVal,yPredSVM)
        for i in range(10):
            f.write("%i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s" \
            % (cm[i,0],'&',cm[i,1],'&',cm[i,2],'&',cm[i,3],'&',cm[i,4],'&',cm[i,5],'&',cm[i,6], \
            '&',cm[i,7],'&',cm[i,8],'&',cm[i,9],'\\\\\n'))

        f.write("\n")
        f.write("%s %1.8f\n" % ("Score Regressao Logistica: ",outparam[0]))
        f.write("%s %4.2f\n" % ("Tempo Regressao Logistica: ",times[0]))

        f.write("\n")
        f.write("Ordenacao dos scores SVM:\n")
        for i in range(nSVM):
            C = dataFrameSVM.at[i,'params']['C']
            solver = dataFrameSVM.at[i,'params']['kernel']
            mScore = dataFrameSVM.at[i,'mean_test_score']
            f.write("%i %s %3.2f %s %s %s %1.4f %s" % (i+1,'&',C,'&',solver,'&',mScore,'\\\\\n'))

        f.write("\n")
        f.write("Matriz de custo SVM:\n")
        cm = skl.metrics.confusion_matrix(yVal,yPredLR)
        for i in range(10):
            f.write("%i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s" \
            % (cm[i,0],'&',cm[i,1],'&',cm[i,2],'&',cm[i,3],'&',cm[i,4],'&',cm[i,5],'&',cm[i,6], \
            '&',cm[i,7],'&',cm[i,8],'&',cm[i,9],'\\\\\n'))

        f.write("\n")
        f.write("%s %1.8f\n" % ("Score SVM: ",outparam[1]))
        f.write("%s %4.2f\n" % ("Tempo SVM: ",times[1]))

        f.write("\n")
        f.write("%s %1.8f\n" % ("Score do modelo final: ",outparam[2]))
        f.write("%s %1.8f\n" % ("Acuracia Ein test: ",outparam[3]))
        f.write("%s %1.8f\n" % ("Acuracia Eout test: ",outparam[4]))
        f.write("%s %1.8f\n" % ("Acuracia Ein validacao: ",outparam[5]))
        f.write("%s %1.8f\n" % ("Precisao Ein test: ",outparam[6]))
        f.write("%s %1.8f\n" % ("Precisao Eout test: ",outparam[7]))
        f.write("%s %1.8f\n" % ("kappa test: ",outparam[8]))        
        f.write("%s %1.8f\n" % ("kappa validacao: ",outparam[9]))  

        f.write("\nMatriz de custo modelo final:\n")
        cm = skl.metrics.confusion_matrix(yTest,testPredFinal)
        for i in range(10):
            f.write("%i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s" \
            % (cm[i,0],'&',cm[i,1],'&',cm[i,2],'&',cm[i,3],'&',cm[i,4],'&',cm[i,5],'&',cm[i,6], \
            '&',cm[i,7],'&',cm[i,8],'&',cm[i,9],'\\\\\n'))

        f.write("\nRe-treinamento:\n")
        f.write("\n")

        f.write("%s %1.8f\n" % ("Score do modelo final: ",outparam[10]))
        f.write("%s %1.8f\n" % ("Acuracia Ein test: ",outparam[11]))
        f.write("%s %1.8f\n" % ("Acuracia Eout test: ",outparam[12]))
        f.write("%s %1.8f\n" % ("Acuracia Ein validacao: ",outparam[13]))
        f.write("%s %1.8f\n" % ("Precisao Ein test: ",outparam[14]))
        f.write("%s %1.8f\n" % ("Precisao Eout test: ",outparam[15]))
        f.write("%s %1.8f\n" % ("kappa test: ",outparam[16]))        
        f.write("%s %1.8f\n" % ("kappa validacao: ",outparam[17])) 

        f.write("\n")
        f.write("Matriz de custo modelo re-treinado:\n")
        cm = skl.metrics.confusion_matrix(yTest,testPredFinalTotal)
        for i in range(10):
            f.write("%i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s %i %s" \
            % (cm[i,0],'&',cm[i,1],'&',cm[i,2],'&',cm[i,3],'&',cm[i,4],'&',cm[i,5],'&',cm[i,6], \
            '&',cm[i,7],'&',cm[i,8],'&',cm[i,9],'\\\\\n'))
        f.write("\n")

    f.close()

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

def test(ntest,trainSize,x_train,y_train,x_test,y_test):
        # Dividindo o conjunto de treinamento para a CV
    print("\nDividindo o conjunto de treinamento para a CV")
    print("---------------------------------------------")
    x_Dtrain,D_val,y_Dtrain,y_Dval,index_Dtrain,index_Dval = train_test_split(x_train,y_train,index,train_size=trainSize,random_state=4,stratify=y_train)

    print("Numero de amostras para treinamento: ",x_Dtrain.shape[0])
    print("Numero de amostras para validacao: ",D_val.shape[0])

    getFrequency(y_Dtrain,y_Dval,kind='Validation')

    if trainSize == 0.80:
        splits = 4
    else:
        splits = 5

    #Logistic Regression
    print("\nAjuste do modelo usando Regressao Logistica")
    print("-------------------------------------------")
    modelLR = LogisticRegression()
    solversLR = ['newton-cg','lbfgs','lbfgs'] 
    rowTableLR = len(solversLR)
    cLR = [100.0,10.0,1.0,0.1,0.01]  
    rowTableLR = rowTableLR * len(cLR)
    gridLR = dict(solver=solversLR,C=cLR,random_state=[4])

    timeLR = time.time()

    gridSearchLR = GridSearchCV(estimator=modelLR, param_grid=gridLR, scoring='accuracy',verbose=3, 
                cv=skl.model_selection.StratifiedKFold(n_splits=splits,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
    gridResultLR = gridSearchLR.fit(x_Dtrain, y_Dtrain) 
    dfLR = pd.DataFrame(gridResultLR.cv_results_)[['params','rank_test_score','mean_test_score']].sort_values(by=['rank_test_score'])

    timeLR = time.time() - timeLR
    times[0] = timeLR

    print("\nTempo de execucao Regressao Logistica: ",int(timeLR),"segundos ou",round(timeLR/60,2),"minutos")

    #SVM
    print("\nAjuste do modelo usando SVM")
    print("---------------------------")
    modelSVM = SVC()
    kernelSVM = ['linear','rbf','sigmoid']
    rowTableSVM = len(kernelSVM)
    cSVM = [100.0,10.0,1.0,0.1,0.01] 
    rowTableSVM = rowTableSVM * len(cSVM)
    gridSVM = dict(kernel=kernelSVM,C=cSVM,random_state=[4])

    timeSVM = time.time()
    gridSearchSVM = GridSearchCV(estimator=modelSVM, param_grid=gridSVM, scoring='accuracy',verbose=3, 
                cv=skl.model_selection.StratifiedKFold(n_splits=splits,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
    gridResultSVM = gridSearchSVM.fit(x_Dtrain, y_Dtrain) 
    dfSVM = pd.DataFrame(gridResultSVM.cv_results_)[['params','rank_test_score','mean_test_score']].sort_values(by=['rank_test_score'])
    timeSVM = time.time() - timeSVM
    times[1] = timeSVM

    print("\nTempo de execucao SVM: ",int(timeSVM),"segundos ou",round(timeSVM/60,2),"minutos")

    yDvalPredLR = gridResultLR.predict(D_val)
    yDvalPredSVM   = gridResultSVM.predict(D_val)

    scoreDvalLR = gridResultLR.score(D_val,y_Dval)
    scoreDvalSVM = gridResultSVM.score(D_val,y_Dval)

    if scoreDvalLR > scoreDvalSVM:
        finalModel = gridResultLR
    else:
        finalModel = gridResultSVM

    finalModel = finalModel.best_estimator_
    print("O melhor modelo e: ", finalModel)

    # Testando o modelo final
    scorexTestFinal = finalModel.score(x_test,y_test)
    testPredictFinal = finalModel.predict(x_test)
    DvalPredictFinal = finalModel.predict(D_val)
    EinTest = skl.metrics.accuracy_score(y_test,testPredictFinal)
    EoutTest = 1 - EinTest
    EinDval = skl.metrics.accuracy_score(y_Dval,DvalPredictFinal)
    EinTestPrecision = skl.metrics.precision_score(y_test,testPredictFinal,average='macro')
    EinDvalPrecision =  skl.metrics.precision_score(y_Dval,DvalPredictFinal,average='macro')
    kappaTest = skl.metrics.recall_score(y_test,testPredictFinal,average='macro')
    kappaDval = skl.metrics.recall_score(y_Dval,DvalPredictFinal,average='macro')

    # Re-treinamento
    finalModelTotal = finalModel.fit(x_train,y_train)
    testPredictFinalTotal = finalModelTotal.predict(x_test)
    DvalPredictFinalTotal = finalModelTotal.predict(D_val)

    scorexTestFinalTotal = finalModelTotal.score(x_test,y_test)
    testPredictFinalTotal = finalModelTotal.predict(x_test)
    DvalPredictFinalTotal = finalModelTotal.predict(D_val)
    EinTestTotal = skl.metrics.accuracy_score(y_test,testPredictFinalTotal)
    EoutTestTotal = 1 - EinTestTotal
    EinDvalTotal = skl.metrics.accuracy_score(y_Dval,DvalPredictFinalTotal)
    EinTestPrecisionTotal = skl.metrics.precision_score(y_test,testPredictFinalTotal,average='macro')
    EinDvalPrecisionTotal =  skl.metrics.precision_score(y_Dval,DvalPredictFinalTotal,average='macro')
    kappaTestTotal = skl.metrics.recall_score(y_test,testPredictFinalTotal,average='macro')
    kappaDvalTotal = skl.metrics.recall_score(y_Dval,DvalPredictFinalTotal,average='macro')

    output[:] = [scoreDvalLR,scoreDvalSVM,scorexTestFinal,EinTest,EoutTest,EinDval,EinTestPrecision,\
                EinDvalPrecision,kappaTest,kappaDval,scorexTestFinalTotal,EinTestTotal,EoutTestTotal,\
                EinDvalTotal,EinTestPrecisionTotal,EinDvalPrecisionTotal,kappaTestTotal,kappaDvalTotal]

    export(ntest,y_Dval,yDvalPredLR,yDvalPredSVM,y_test,testPredictFinal,testPredictFinalTotal,dfLR,dfSVM,rowTableLR,rowTableSVM,output)
    

totalTime = time.time()
# Carregamento dos dados
print("\nCarregando o dataset: Fashion mnist")
print("-----------------------------------")
(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()

xTrainReduced = np.array([image[::2, 1::2] for image in xTrain])
xTestReduced = np.array([image[::2, 1::2] for image in xTest])

print("Numero de amostras para treinamento: ",xTrain.shape[0])
print("Numero de amostras para teste: ",xTest.shape[0])
print("Tamanho de cada amostra: ",xTrain[0,:,:].shape," pixels de escala de cinza")
print("Tamanho de cada amostra reduzida: ",xTrainReduced[0,:,:].shape," pixels de escala de cinza")

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

getFrequency(yTrain,yTest,kind='Test')
getImages(xTrain,yTrain)

# Normalizacao
xTrain = (xTrain/255.0).astype('float32').reshape((60000,28*28))
xTest = (xTest/255.0).astype('float32').reshape((10000,28*28))

xTrainReduced = (xTrainReduced/255.0).astype('float32').reshape((60000,14*14))
xTestReduced = (xTestReduced/255.0).astype('float32').reshape((10000,14*14))

N, d = xTrain.shape
index = np.arange(N)
output = np.empty(18)
times = np.zeros(2)
testTimes = np.zeros(4)

testTimes[0] = time.time()
print("\n-------------------")
print("Execucao do teste 1")
print("-------------------")
test(1,0.80,xTrain,yTrain,xTest,yTest)
testTimes[0] = time.time() - testTimes[0]

testTimes[1] = time.time()
print("\n-------------------")
print("Execucao do teste 2")
print("-------------------")
test(2,0.75,xTrain,yTrain,xTest,yTest)
testTimes[1] = time.time() - testTimes[1]

testTimes[2] = time.time()
print("\n-------------------")
print("Execucao do teste 3")
print("-------------------")
test(3,0.80,xTrainReduced,yTrain,xTestReduced,yTest)
testTimes[2] = time.time() - testTimes[2]

testTimes[3] = time.time()
print("\n-------------------")
print("Execucao do teste 4")
print("-------------------")
test(4,0.75,xTrainReduced,yTrain,xTestReduced,yTest)
testTimes[3] = time.time() - testTimes[3]

totalTime = time.time() - totalTime

print("Tempo total: ",round(totalTime/60,2)," minutos ou",round(totalTime/3600,2)," horas.")

with open('totalTime','w') as f:
    f.write("%s %4.2f %s" % ("Tempo do teste 1: ",testTimes[0],"\n"))
    f.write("%s %4.2f %s" % ("Tempo do teste 2: ",testTimes[1],"\n"))
    f.write("%s %4.2f %s" % ("Tempo do teste 3: ",testTimes[2],"\n"))
    f.write("%s %4.2f %s" % ("Tempo do teste 4: ",testTimes[3],"\n"))
    f.write("%s %4.2f" % ("Tempo total de todos os testes: ",totalTime))

f.close()