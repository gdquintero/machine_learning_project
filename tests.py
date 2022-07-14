#!/usr/bin/env python
# coding: utf-8

def test1():
        # Dividindo o conjunto de treinamento para a CV
    print("\nDividindo o conjunto de treinamento para a CV")
    print("---------------------------------------------")
    x_Dtrain,D_val,y_Dtrain,y_Dval,index_Dtrain,index_Dval = train_test_split(x_train,y_train,index,train_size=0.80,random_state=4,stratify=y_train)

    print("Numero de amostras para treinamento: ",x_Dtrain.shape[0])
    print("Numero de amostras para validacao: ",D_val.shape[0])

    getFrequency(y_Dtrain,y_Dval,kind='Validation')

    #Logistic Regression
    print("\nAjuste do modelo usando Regressao Logistica")
    print("-------------------------------------------")
    model_lgreg = LogisticRegression()
    solvers_lgreg = ['lbfgs']
    c_values_lgreg = [0.01] 
    grid_lgreg = dict(solver=solvers_lgreg,C=c_values_lgreg,random_state=[4])

    time_lgreg = time.time()

    grid_search_lgreg = GridSearchCV(estimator=model_lgreg, param_grid=grid_lgreg, scoring='accuracy',verbose=3, 
                cv=skl.model_selection.StratifiedKFold(n_splits=2,random_state=4,shuffle=True).split(x_Dtrain,y_Dtrain))
    grid_result_lgreg = grid_search_lgreg.fit(x_Dtrain, y_Dtrain) 

    time_lgreg = time.time() - time_lgreg

    # pd.DataFrame(grid_result_lgreg.cv_results_)[['params','rank_test_score','mean_test_score']].sort_values(by=['rank_test_score'])

    print("\nTempo de execucao: ",int(time_lgreg),"segundos ou",round(time_lgreg/60,2),"minutos")