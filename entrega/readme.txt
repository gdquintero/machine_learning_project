Projeto final Machine Learning (MAC5832)
Dayanne Gomes, NUSP: 13796211
Gustavo Álvarez, NUSP: 11350395

Neste arquivo encontra-se informação sobre as versões de python usadas e algumas bibliotecas 
relevantes na execução dos programas principais main.py (clases balanceadas) e 
main_unbalanced.py (classes desbalanceadas). Também é mostrado a saída no terminal após a execução
de cada programa.

Versao do Python:  3.8.13
Versao da biblioteca Numpy:  1.22.4
Versao da biblioteca Matplotlib:  1.22.4
Versao da biblioteca Pandas:  1.4.2
Versao da biblioteca Tensorflow:  2.9.1
Versao da biblioteca Scikit-learn:  1.1.1

Se deseja executar os testes 1, 2 , 3 e 4 usando classes balanceadas, execute o seguinte comando 
python3 main.py na sua terminal (usando por exemplo um ambiente virtual com as versões explicitadas acima). 
Se deseja executar os testes 1 e 2 usando clases desbalanceadas execute o comando python3 main_unbalanced.py.

Saída dos dois programas:

------------------
Inicio do programa
------------------

Carregando o dataset: Fashion mnist
-----------------------------------
Numero de amostras para treinamento:  60000
Numero de amostras para teste:  10000
Tamanho de cada amostra:  (28, 28)  pixels de escala de cinza
Tamanho de cada amostra reduzida:  (14, 14)  pixels de escala de cinza

**********************************************************************
CLASSES BALANCEADAS
**********************************************************************

-------------------
Execucao do teste 1
-------------------

Dividindo o conjunto de treinamento para a CV
---------------------------------------------
Numero de amostras para treinamento:  48000
Numero de amostras para validacao:  12000

Ajuste do modelo usando Regressao Logistica
-------------------------------------------
Fitting 4 folds for each of 10 candidates, totalling 40 fits
[CV 1/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.832 total time= 2.6min
[CV 2/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.830 total time= 4.4min
[CV 3/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.829 total time= 4.2min
[CV 4/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.831 total time= 4.2min
[CV 1/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.850 total time=   7.7s
[CV 2/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.857 total time=   7.6s
[CV 3/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.851 total time=   7.4s
[CV 4/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.852 total time=   7.8s
[CV 1/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.840 total time= 1.9min
[CV 2/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.841 total time= 1.9min
[CV 3/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.839 total time= 1.9min
[CV 4/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.839 total time= 2.0min
[CV 1/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.852 total time=   7.7s
[CV 2/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.856 total time=   7.6s
[CV 3/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.853 total time=   7.4s
[CV 4/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.852 total time=   7.7s
[CV 1/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.849 total time=  47.1s
[CV 2/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.851 total time=  50.5s
[CV 3/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.849 total time=  54.1s
[CV 4/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.847 total time= 1.1min
[CV 1/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.852 total time=   7.7s
[CV 2/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.859 total time=   7.8s
[CV 3/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.853 total time=   7.4s
[CV 4/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.851 total time=   7.3s
[CV 1/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.856 total time=  26.2s
[CV 2/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.860 total time=  22.9s
[CV 3/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.855 total time=  26.8s
[CV 4/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.850 total time=  34.3s
[CV 1/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.854 total time=   7.5s
[CV 2/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.860 total time=   7.7s
[CV 3/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.854 total time=   7.5s
[CV 4/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.852 total time=   7.2s
[CV 1/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.847 total time=  17.7s
[CV 2/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.854 total time=  24.6s
[CV 3/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.848 total time=  16.5s
[CV 4/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.844 total time=  16.8s
[CV 1/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.847 total time=   7.6s
[CV 2/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.854 total time=   7.5s
[CV 3/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.850 total time=   7.9s
[CV 4/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.845 total time=   7.6s

Tempo de execucao Regressao Logistica:  1980 segundos ou 33.0 minutos

Ajuste do modelo usando SVM
---------------------------
Fitting 4 folds for each of 10 candidates, totalling 40 fits
[CV 1/4] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time=26.6min
[CV 2/4] END C=100.0, kernel=linear, random_state=4;, score=0.830 total time=27.1min
[CV 3/4] END C=100.0, kernel=linear, random_state=4;, score=0.827 total time=26.5min
[CV 4/4] END C=100.0, kernel=linear, random_state=4;, score=0.831 total time=27.3min
[CV 1/4] END C=100.0, kernel=rbf, random_state=4;, score=0.893 total time= 2.3min
[CV 2/4] END C=100.0, kernel=rbf, random_state=4;, score=0.895 total time= 2.3min
[CV 3/4] END C=100.0, kernel=rbf, random_state=4;, score=0.888 total time= 2.3min
[CV 4/4] END C=100.0, kernel=rbf, random_state=4;, score=0.891 total time= 2.3min
[CV 1/4] END C=10.0, kernel=linear, random_state=4;, score=0.839 total time= 3.7min
[CV 2/4] END C=10.0, kernel=linear, random_state=4;, score=0.835 total time= 3.8min
[CV 3/4] END C=10.0, kernel=linear, random_state=4;, score=0.834 total time= 3.9min
[CV 4/4] END C=10.0, kernel=linear, random_state=4;, score=0.839 total time= 3.8min
[CV 1/4] END C=10.0, kernel=rbf, random_state=4;, score=0.899 total time= 2.2min
[CV 2/4] END C=10.0, kernel=rbf, random_state=4;, score=0.905 total time= 2.2min
[CV 3/4] END C=10.0, kernel=rbf, random_state=4;, score=0.897 total time= 2.2min
[CV 4/4] END C=10.0, kernel=rbf, random_state=4;, score=0.898 total time= 2.3min
[CV 1/4] END C=1.0, kernel=linear, random_state=4;, score=0.852 total time= 1.8min
[CV 2/4] END C=1.0, kernel=linear, random_state=4;, score=0.850 total time= 1.8min
[CV 3/4] END C=1.0, kernel=linear, random_state=4;, score=0.848 total time= 1.8min
[CV 4/4] END C=1.0, kernel=linear, random_state=4;, score=0.848 total time= 1.8min
[CV 1/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.885 total time= 2.4min
[CV 2/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.886 total time= 2.5min
[CV 3/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.882 total time= 2.4min
[CV 4/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.883 total time= 2.4min
[CV 1/4] END C=0.1, kernel=linear, random_state=4;, score=0.862 total time= 1.6min
[CV 2/4] END C=0.1, kernel=linear, random_state=4;, score=0.866 total time= 1.6min
[CV 3/4] END C=0.1, kernel=linear, random_state=4;, score=0.861 total time= 1.6min
[CV 4/4] END C=0.1, kernel=linear, random_state=4;, score=0.860 total time= 1.6min
[CV 1/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.842 total time= 3.6min
[CV 2/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.846 total time= 3.6min
[CV 3/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.843 total time= 3.6min
[CV 4/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.838 total time= 3.6min
[CV 1/4] END C=0.01, kernel=linear, random_state=4;, score=0.860 total time= 1.9min
[CV 2/4] END C=0.01, kernel=linear, random_state=4;, score=0.863 total time= 1.9min
[CV 3/4] END C=0.01, kernel=linear, random_state=4;, score=0.860 total time= 1.9min
[CV 4/4] END C=0.01, kernel=linear, random_state=4;, score=0.858 total time= 1.9min
[CV 1/4] END C=0.01, kernel=rbf, random_state=4;, score=0.767 total time= 6.9min
[CV 2/4] END C=0.01, kernel=rbf, random_state=4;, score=0.771 total time= 6.9min
[CV 3/4] END C=0.01, kernel=rbf, random_state=4;, score=0.765 total time= 6.9min
[CV 4/4] END C=0.01, kernel=rbf, random_state=4;, score=0.764 total time= 6.9min

Tempo de execucao SVM:  12913 segundos ou 215.23 minutos
O melhor modelo e:  SVC(C=10.0, random_state=4)

-------------------
Execucao do teste 2
-------------------

Dividindo o conjunto de treinamento para a CV
---------------------------------------------
Numero de amostras para treinamento:  45000
Numero de amostras para validacao:  15000

Ajuste do modelo usando Regressao Logistica
-------------------------------------------
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[CV 1/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.827 total time= 5.9min
[CV 2/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.830 total time= 5.0min
[CV 3/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.834 total time= 4.5min
[CV 4/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.833 total time= 3.5min
[CV 5/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.836 total time= 6.0min
[CV 1/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.850 total time=   7.3s
[CV 2/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.851 total time=   7.6s
[CV 3/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.858 total time=   7.4s
[CV 4/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.855 total time=   7.6s
[CV 5/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.857 total time=   7.6s
[CV 1/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.834 total time= 1.9min
[CV 2/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.838 total time= 1.7min
[CV 3/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.844 total time= 2.1min
[CV 4/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.840 total time= 2.3min
[CV 5/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.842 total time= 2.5min
[CV 1/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.849 total time=   7.6s
[CV 2/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.851 total time=   8.0s
[CV 3/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.857 total time=   7.8s
[CV 4/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.855 total time=   7.4s
[CV 5/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.858 total time=   7.4s
[CV 1/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.844 total time=  51.9s
[CV 2/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.849 total time=  49.5s
[CV 3/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.854 total time=  50.7s
[CV 4/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.847 total time=  48.0s
[CV 5/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.853 total time= 1.0min
[CV 1/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.849 total time=   7.4s
[CV 2/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.852 total time=   7.4s
[CV 3/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.858 total time=   7.6s
[CV 4/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.853 total time=   7.6s
[CV 5/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.856 total time=   7.5s
[CV 1/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.853 total time=  23.6s
[CV 2/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.853 total time=  25.9s
[CV 3/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.858 total time=  34.7s
[CV 4/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.852 total time=  26.9s
[CV 5/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.858 total time=  23.1s
[CV 1/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.853 total time=   7.4s
[CV 2/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.852 total time=   7.5s
[CV 3/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.857 total time=   7.6s
[CV 4/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.855 total time=   7.8s
[CV 5/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.859 total time=   7.3s
[CV 1/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.844 total time=  14.8s
[CV 2/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.843 total time=  23.2s
[CV 3/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.851 total time=  15.6s
[CV 4/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.845 total time=  15.7s
[CV 5/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.854 total time=  20.5s
[CV 1/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.845 total time=   7.5s
[CV 2/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.845 total time=   7.6s
[CV 3/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.850 total time=   7.6s
[CV 4/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.845 total time=   7.2s
[CV 5/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.855 total time=   7.8s

Tempo de execucao Regressao Logistica:  2807 segundos ou 46.8 minutos

Ajuste do modelo usando SVM
---------------------------
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[CV 1/5] END C=100.0, kernel=linear, random_state=4;, score=0.822 total time=25.9min
[CV 2/5] END C=100.0, kernel=linear, random_state=4;, score=0.830 total time=27.7min
[CV 3/5] END C=100.0, kernel=linear, random_state=4;, score=0.832 total time=28.0min
[CV 4/5] END C=100.0, kernel=linear, random_state=4;, score=0.834 total time=27.7min
[CV 5/5] END C=100.0, kernel=linear, random_state=4;, score=0.833 total time=26.0min
[CV 1/5] END C=100.0, kernel=rbf, random_state=4;, score=0.888 total time= 2.0min
[CV 2/5] END C=100.0, kernel=rbf, random_state=4;, score=0.894 total time= 2.0min
[CV 3/5] END C=100.0, kernel=rbf, random_state=4;, score=0.894 total time= 2.0min
[CV 4/5] END C=100.0, kernel=rbf, random_state=4;, score=0.893 total time= 2.0min
[CV 5/5] END C=100.0, kernel=rbf, random_state=4;, score=0.892 total time= 2.0min
[CV 1/5] END C=10.0, kernel=linear, random_state=4;, score=0.832 total time= 3.7min
[CV 2/5] END C=10.0, kernel=linear, random_state=4;, score=0.835 total time= 3.6min
[CV 3/5] END C=10.0, kernel=linear, random_state=4;, score=0.837 total time= 3.8min
[CV 4/5] END C=10.0, kernel=linear, random_state=4;, score=0.841 total time= 3.7min
[CV 5/5] END C=10.0, kernel=linear, random_state=4;, score=0.843 total time= 3.8min
[CV 1/5] END C=10.0, kernel=rbf, random_state=4;, score=0.894 total time= 1.9min
[CV 2/5] END C=10.0, kernel=rbf, random_state=4;, score=0.899 total time= 1.9min
[CV 3/5] END C=10.0, kernel=rbf, random_state=4;, score=0.903 total time= 2.0min
[CV 4/5] END C=10.0, kernel=rbf, random_state=4;, score=0.902 total time= 2.0min
[CV 5/5] END C=10.0, kernel=rbf, random_state=4;, score=0.902 total time= 2.0min
[CV 1/5] END C=1.0, kernel=linear, random_state=4;, score=0.847 total time= 1.7min
[CV 2/5] END C=1.0, kernel=linear, random_state=4;, score=0.848 total time= 1.7min
[CV 3/5] END C=1.0, kernel=linear, random_state=4;, score=0.856 total time= 1.7min
[CV 4/5] END C=1.0, kernel=linear, random_state=4;, score=0.854 total time= 1.7min
[CV 5/5] END C=1.0, kernel=linear, random_state=4;, score=0.854 total time= 1.7min
[CV 1/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.882 total time= 2.2min
[CV 2/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.880 total time= 2.1min
[CV 3/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.888 total time= 2.1min
[CV 4/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.885 total time= 2.1min
[CV 5/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.888 total time= 2.2min
[CV 1/5] END C=0.1, kernel=linear, random_state=4;, score=0.857 total time= 1.5min
[CV 2/5] END C=0.1, kernel=linear, random_state=4;, score=0.862 total time= 1.5min
[CV 3/5] END C=0.1, kernel=linear, random_state=4;, score=0.870 total time= 1.5min
[CV 4/5] END C=0.1, kernel=linear, random_state=4;, score=0.864 total time= 1.5min
[CV 5/5] END C=0.1, kernel=linear, random_state=4;, score=0.866 total time= 1.5min
[CV 1/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.843 total time= 3.3min
[CV 2/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.831 total time= 3.3min
[CV 3/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.841 total time= 3.3min
[CV 4/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.840 total time= 3.3min
[CV 5/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.848 total time= 3.2min
[CV 1/5] END C=0.01, kernel=linear, random_state=4;, score=0.855 total time= 1.7min
[CV 2/5] END C=0.01, kernel=linear, random_state=4;, score=0.854 total time= 1.7min
[CV 3/5] END C=0.01, kernel=linear, random_state=4;, score=0.865 total time= 1.7min
[CV 4/5] END C=0.01, kernel=linear, random_state=4;, score=0.859 total time= 1.7min
[CV 5/5] END C=0.01, kernel=linear, random_state=4;, score=0.865 total time= 1.7min
[CV 1/5] END C=0.01, kernel=rbf, random_state=4;, score=0.765 total time= 6.3min
[CV 2/5] END C=0.01, kernel=rbf, random_state=4;, score=0.758 total time= 6.3min
[CV 3/5] END C=0.01, kernel=rbf, random_state=4;, score=0.768 total time= 6.3min
[CV 4/5] END C=0.01, kernel=rbf, random_state=4;, score=0.768 total time= 6.3min
[CV 5/5] END C=0.01, kernel=rbf, random_state=4;, score=0.773 total time= 6.3min

Tempo de execucao SVM:  15481 segundos ou 258.03 minutos
O melhor modelo e:  SVC(C=10.0, random_state=4)

-------------------
Execucao do teste 3
-------------------

Dividindo o conjunto de treinamento para a CV
---------------------------------------------
Numero de amostras para treinamento:  48000
Numero de amostras para validacao:  12000

Ajuste do modelo usando Regressao Logistica
-------------------------------------------
Fitting 4 folds for each of 10 candidates, totalling 40 fits
[CV 1/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.825 total time=  49.2s
[CV 2/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.829 total time= 1.3min
[CV 3/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.826 total time= 1.4min
[CV 4/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.823 total time=  50.7s
[CV 1/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.8s
[CV 2/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.830 total time=   2.7s
[CV 3/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.825 total time=   2.7s
[CV 4/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 1/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.826 total time=  23.4s
[CV 2/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.830 total time=  27.1s
[CV 3/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.827 total time=  19.0s
[CV 4/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.825 total time=  27.8s
[CV 1/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.8s
[CV 2/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.8s
[CV 3/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 4/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.7s
[CV 1/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.825 total time=   9.9s
[CV 2/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.833 total time=  10.2s
[CV 3/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.828 total time=  10.1s
[CV 4/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.825 total time=  10.2s
[CV 1/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 2/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.8s
[CV 3/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.826 total time=   2.8s
[CV 4/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.825 total time=   2.9s
[CV 1/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.823 total time=   6.2s
[CV 2/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.831 total time=   6.4s
[CV 3/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.825 total time=   6.6s
[CV 4/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.821 total time=   8.1s
[CV 1/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.822 total time=   2.7s
[CV 2/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.831 total time=   2.7s
[CV 3/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 4/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.820 total time=   2.8s
[CV 1/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.803 total time=   7.1s
[CV 2/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.812 total time=   5.0s
[CV 3/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.804 total time=   4.9s
[CV 4/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.802 total time=   4.9s
[CV 1/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.803 total time=   2.8s
[CV 2/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.812 total time=   2.8s
[CV 3/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.804 total time=   2.8s
[CV 4/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.802 total time=   2.8s

Tempo de execucao Regressao Logistica:  519 segundos ou 8.67 minutos

Ajuste do modelo usando SVM
---------------------------
Fitting 4 folds for each of 10 candidates, totalling 40 fits
[CV 1/4] END C=100.0, kernel=linear, random_state=4;, score=0.832 total time= 5.2min
[CV 2/4] END C=100.0, kernel=linear, random_state=4;, score=0.835 total time= 5.4min
[CV 3/4] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.4min
[CV 4/4] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.3min
[CV 1/4] END C=100.0, kernel=rbf, random_state=4;, score=0.872 total time=  54.5s
[CV 2/4] END C=100.0, kernel=rbf, random_state=4;, score=0.879 total time=  54.0s
[CV 3/4] END C=100.0, kernel=rbf, random_state=4;, score=0.872 total time=  54.2s
[CV 4/4] END C=100.0, kernel=rbf, random_state=4;, score=0.877 total time=  54.6s
[CV 1/4] END C=10.0, kernel=linear, random_state=4;, score=0.834 total time= 1.2min
[CV 2/4] END C=10.0, kernel=linear, random_state=4;, score=0.838 total time= 1.2min
[CV 3/4] END C=10.0, kernel=linear, random_state=4;, score=0.830 total time= 1.2min
[CV 4/4] END C=10.0, kernel=linear, random_state=4;, score=0.829 total time= 1.2min
[CV 1/4] END C=10.0, kernel=rbf, random_state=4;, score=0.882 total time=  50.8s
[CV 2/4] END C=10.0, kernel=rbf, random_state=4;, score=0.882 total time=  51.3s
[CV 3/4] END C=10.0, kernel=rbf, random_state=4;, score=0.880 total time=  50.5s
[CV 4/4] END C=10.0, kernel=rbf, random_state=4;, score=0.878 total time=  51.0s
[CV 1/4] END C=1.0, kernel=linear, random_state=4;, score=0.836 total time=  39.1s
[CV 2/4] END C=1.0, kernel=linear, random_state=4;, score=0.841 total time=  38.7s
[CV 3/4] END C=1.0, kernel=linear, random_state=4;, score=0.836 total time=  38.9s
[CV 4/4] END C=1.0, kernel=linear, random_state=4;, score=0.833 total time=  39.0s
[CV 1/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.864 total time=  57.0s
[CV 2/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.868 total time=  56.3s
[CV 3/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.865 total time=  56.2s
[CV 4/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.864 total time=  57.4s
[CV 1/4] END C=0.1, kernel=linear, random_state=4;, score=0.834 total time=  37.7s
[CV 2/4] END C=0.1, kernel=linear, random_state=4;, score=0.841 total time=  37.6s
[CV 3/4] END C=0.1, kernel=linear, random_state=4;, score=0.834 total time=  37.7s
[CV 4/4] END C=0.1, kernel=linear, random_state=4;, score=0.831 total time=  37.7s
[CV 1/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.821 total time= 1.4min
[CV 2/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.828 total time= 1.4min
[CV 3/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.824 total time= 1.4min
[CV 4/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.821 total time= 1.4min
[CV 1/4] END C=0.01, kernel=linear, random_state=4;, score=0.817 total time=  49.9s
[CV 2/4] END C=0.01, kernel=linear, random_state=4;, score=0.823 total time=  50.1s
[CV 3/4] END C=0.01, kernel=linear, random_state=4;, score=0.819 total time=  50.1s
[CV 4/4] END C=0.01, kernel=linear, random_state=4;, score=0.816 total time=  50.1s
[CV 1/4] END C=0.01, kernel=rbf, random_state=4;, score=0.758 total time= 2.7min
[CV 2/4] END C=0.01, kernel=rbf, random_state=4;, score=0.761 total time= 2.7min
[CV 3/4] END C=0.01, kernel=rbf, random_state=4;, score=0.756 total time= 2.7min
[CV 4/4] END C=0.01, kernel=rbf, random_state=4;, score=0.756 total time= 2.7min

Tempo de execucao SVM:  3732 segundos ou 62.21 minutos
O melhor modelo e:  SVC(C=10.0, random_state=4)

-------------------
Execucao do teste 4
-------------------

Dividindo o conjunto de treinamento para a CV
---------------------------------------------
Numero de amostras para treinamento:  45000
Numero de amostras para validacao:  15000

Ajuste do modelo usando Regressao Logistica
-------------------------------------------
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[CV 1/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.826 total time=  51.0s
[CV 2/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.821 total time= 1.1min
[CV 3/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.829 total time=  27.8s
[CV 4/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.824 total time= 1.5min
[CV 5/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.827 total time=  58.4s
[CV 1/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.8s
[CV 2/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.821 total time=   2.7s
[CV 3/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.7s
[CV 4/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.8s
[CV 5/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.7s
[CV 1/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.827 total time=  32.2s
[CV 2/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.821 total time=  34.6s
[CV 3/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.831 total time=  42.7s
[CV 4/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.824 total time=  25.3s
[CV 5/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.828 total time=  21.1s
[CV 1/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 2/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.7s
[CV 3/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.826 total time=   2.8s
[CV 4/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.821 total time=   2.7s
[CV 5/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.7s
[CV 1/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.827 total time=  16.2s
[CV 2/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.822 total time=  10.9s
[CV 3/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.830 total time=  10.9s
[CV 4/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.825 total time=  12.7s
[CV 5/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.831 total time=  12.9s
[CV 1/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.7s
[CV 2/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.8s
[CV 3/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.826 total time=   2.8s
[CV 4/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.9s
[CV 5/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.830 total time=   2.7s
[CV 1/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.823 total time=   8.5s
[CV 2/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.820 total time=   5.6s
[CV 3/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.830 total time=   6.6s
[CV 4/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.824 total time=   5.5s
[CV 5/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.829 total time=   8.0s
[CV 1/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.822 total time=   2.7s
[CV 2/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.820 total time=   2.7s
[CV 3/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.830 total time=   2.8s
[CV 4/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.823 total time=   2.7s
[CV 5/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.829 total time=   2.8s
[CV 1/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.802 total time=   4.4s
[CV 2/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.801 total time=   8.9s
[CV 3/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.807 total time=   6.6s
[CV 4/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.806 total time=   4.5s
[CV 5/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.811 total time=   4.1s
[CV 1/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.802 total time=   2.8s
[CV 2/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.800 total time=   2.8s
[CV 3/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.807 total time=   2.7s
[CV 4/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.806 total time=   2.8s
[CV 5/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.812 total time=   2.8s

Tempo de execucao Regressao Logistica:  660 segundos ou 11.01 minutos

Ajuste do modelo usando SVM
---------------------------
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[CV 1/5] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.3min
[CV 2/5] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.3min
[CV 3/5] END C=100.0, kernel=linear, random_state=4;, score=0.834 total time= 5.5min
[CV 4/5] END C=100.0, kernel=linear, random_state=4;, score=0.833 total time= 5.2min
[CV 5/5] END C=100.0, kernel=linear, random_state=4;, score=0.833 total time= 5.5min
[CV 1/5] END C=100.0, kernel=rbf, random_state=4;, score=0.873 total time=  47.9s
[CV 2/5] END C=100.0, kernel=rbf, random_state=4;, score=0.873 total time=  48.1s
[CV 3/5] END C=100.0, kernel=rbf, random_state=4;, score=0.876 total time=  48.5s
[CV 4/5] END C=100.0, kernel=rbf, random_state=4;, score=0.876 total time=  48.6s
[CV 5/5] END C=100.0, kernel=rbf, random_state=4;, score=0.876 total time=  48.4s
[CV 1/5] END C=10.0, kernel=linear, random_state=4;, score=0.830 total time= 1.1min
[CV 2/5] END C=10.0, kernel=linear, random_state=4;, score=0.831 total time= 1.1min
[CV 3/5] END C=10.0, kernel=linear, random_state=4;, score=0.835 total time= 1.1min
[CV 4/5] END C=10.0, kernel=linear, random_state=4;, score=0.834 total time= 1.1min
[CV 5/5] END C=10.0, kernel=linear, random_state=4;, score=0.835 total time= 1.1min
[CV 1/5] END C=10.0, kernel=rbf, random_state=4;, score=0.878 total time=  44.0s
[CV 2/5] END C=10.0, kernel=rbf, random_state=4;, score=0.879 total time=  45.1s
[CV 3/5] END C=10.0, kernel=rbf, random_state=4;, score=0.881 total time=  44.6s
[CV 4/5] END C=10.0, kernel=rbf, random_state=4;, score=0.885 total time=  45.3s
[CV 5/5] END C=10.0, kernel=rbf, random_state=4;, score=0.881 total time=  44.8s
[CV 1/5] END C=1.0, kernel=linear, random_state=4;, score=0.833 total time=  35.7s
[CV 2/5] END C=1.0, kernel=linear, random_state=4;, score=0.832 total time=  35.6s
[CV 3/5] END C=1.0, kernel=linear, random_state=4;, score=0.837 total time=  36.0s
[CV 4/5] END C=1.0, kernel=linear, random_state=4;, score=0.835 total time=  35.9s
[CV 5/5] END C=1.0, kernel=linear, random_state=4;, score=0.838 total time=  35.7s
[CV 1/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.862 total time=  49.0s
[CV 2/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.862 total time=  48.6s
[CV 3/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.865 total time=  49.0s
[CV 4/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.865 total time=  50.0s
[CV 5/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.870 total time=  50.2s
[CV 1/5] END C=0.1, kernel=linear, random_state=4;, score=0.834 total time=  34.1s
[CV 2/5] END C=0.1, kernel=linear, random_state=4;, score=0.832 total time=  34.0s
[CV 3/5] END C=0.1, kernel=linear, random_state=4;, score=0.838 total time=  34.3s
[CV 4/5] END C=0.1, kernel=linear, random_state=4;, score=0.833 total time=  34.1s
[CV 5/5] END C=0.1, kernel=linear, random_state=4;, score=0.837 total time=  34.2s
[CV 1/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.823 total time= 1.2min
[CV 2/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.817 total time= 1.2min
[CV 3/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.822 total time= 1.2min
[CV 4/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.819 total time= 1.2min
[CV 5/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.829 total time= 1.2min
[CV 1/5] END C=0.01, kernel=linear, random_state=4;, score=0.813 total time=  45.7s
[CV 2/5] END C=0.01, kernel=linear, random_state=4;, score=0.817 total time=  45.6s
[CV 3/5] END C=0.01, kernel=linear, random_state=4;, score=0.822 total time=  46.0s
[CV 4/5] END C=0.01, kernel=linear, random_state=4;, score=0.813 total time=  45.7s
[CV 5/5] END C=0.01, kernel=linear, random_state=4;, score=0.825 total time=  45.8s
[CV 1/5] END C=0.01, kernel=rbf, random_state=4;, score=0.755 total time= 2.4min
[CV 2/5] END C=0.01, kernel=rbf, random_state=4;, score=0.752 total time= 2.4min
[CV 3/5] END C=0.01, kernel=rbf, random_state=4;, score=0.758 total time= 2.4min
[CV 4/5] END C=0.01, kernel=rbf, random_state=4;, score=0.756 total time= 2.4min
[CV 5/5] END C=0.01, kernel=rbf, random_state=4;, score=0.764 total time= 2.4min

Tempo de execucao SVM:  4355 segundos ou 72.59 minutos
O melhor modelo e:  SVC(C=10.0, random_state=4)
Tempo total:  758.08  minutos ou 12.63  horas.

**********************************************************************
CLASSES DESBALANCEADAS
**********************************************************************

Numero de amostras reduzidas para treinamento:  60000
Numero de amostras para teste:  10000
Tamanho de cada amostra:  (14, 14)  pixels de escala de cinza

-------------------
Execucao do teste 1
-------------------

Dividindo o conjunto de treinamento para a CV
---------------------------------------------
Numero de amostras para treinamento:  48000
Numero de amostras para validacao:  12000

Ajuste do modelo usando Regressao Logistica
-------------------------------------------
Fitting 4 folds for each of 10 candidates, totalling 40 fits
[CV 1/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.825 total time=  48.1s
[CV 2/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.829 total time= 1.3min
[CV 3/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.826 total time= 1.4min
[CV 4/4] END C=100.0, random_state=4, solver=newton-cg;, score=0.823 total time=  50.1s
[CV 1/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.8s
[CV 2/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.830 total time=   2.7s
[CV 3/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.825 total time=   2.7s
[CV 4/4] END C=100.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 1/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.826 total time=  23.1s
[CV 2/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.830 total time=  26.6s
[CV 3/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.827 total time=  18.6s
[CV 4/4] END C=10.0, random_state=4, solver=newton-cg;, score=0.825 total time=  27.2s
[CV 1/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.8s
[CV 2/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.8s
[CV 3/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 4/4] END C=10.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 1/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.825 total time=   9.7s
[CV 2/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.833 total time=  10.1s
[CV 3/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.828 total time=   9.9s
[CV 4/4] END C=1.0, random_state=4, solver=newton-cg;, score=0.825 total time=  10.0s
[CV 1/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 2/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.8s
[CV 3/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.826 total time=   2.8s
[CV 4/4] END C=1.0, random_state=4, solver=lbfgs;, score=0.825 total time=   2.9s
[CV 1/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.823 total time=   6.1s
[CV 2/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.831 total time=   6.3s
[CV 3/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.825 total time=   6.4s
[CV 4/4] END C=0.1, random_state=4, solver=newton-cg;, score=0.821 total time=   7.9s
[CV 1/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.822 total time=   2.8s
[CV 2/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.831 total time=   2.7s
[CV 3/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 4/4] END C=0.1, random_state=4, solver=lbfgs;, score=0.820 total time=   2.8s
[CV 1/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.803 total time=   7.1s
[CV 2/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.812 total time=   4.9s
[CV 3/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.804 total time=   4.9s
[CV 4/4] END C=0.01, random_state=4, solver=newton-cg;, score=0.802 total time=   4.8s
[CV 1/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.803 total time=   2.8s
[CV 2/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.812 total time=   2.9s
[CV 3/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.804 total time=   2.8s
[CV 4/4] END C=0.01, random_state=4, solver=lbfgs;, score=0.802 total time=   2.8s

Tempo de execucao Regressao Logistica:  515 segundos ou 8.59 minutos

Ajuste do modelo usando SVM
---------------------------
Fitting 4 folds for each of 10 candidates, totalling 40 fits
[CV 1/4] END C=100.0, kernel=linear, random_state=4;, score=0.832 total time= 5.3min
[CV 2/4] END C=100.0, kernel=linear, random_state=4;, score=0.835 total time= 5.4min
[CV 3/4] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.4min
[CV 4/4] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.3min
[CV 1/4] END C=100.0, kernel=rbf, random_state=4;, score=0.872 total time=  54.5s
[CV 2/4] END C=100.0, kernel=rbf, random_state=4;, score=0.879 total time=  54.0s
[CV 3/4] END C=100.0, kernel=rbf, random_state=4;, score=0.872 total time=  55.0s
[CV 4/4] END C=100.0, kernel=rbf, random_state=4;, score=0.877 total time=  55.9s
[CV 1/4] END C=10.0, kernel=linear, random_state=4;, score=0.834 total time= 1.1min
[CV 2/4] END C=10.0, kernel=linear, random_state=4;, score=0.838 total time= 1.1min
[CV 3/4] END C=10.0, kernel=linear, random_state=4;, score=0.830 total time= 1.1min
[CV 4/4] END C=10.0, kernel=linear, random_state=4;, score=0.829 total time= 1.1min
[CV 1/4] END C=10.0, kernel=rbf, random_state=4;, score=0.882 total time=  52.7s
[CV 2/4] END C=10.0, kernel=rbf, random_state=4;, score=0.882 total time=  51.9s
[CV 3/4] END C=10.0, kernel=rbf, random_state=4;, score=0.880 total time=  51.7s
[CV 4/4] END C=10.0, kernel=rbf, random_state=4;, score=0.878 total time=  52.0s
[CV 1/4] END C=1.0, kernel=linear, random_state=4;, score=0.836 total time=  38.4s
[CV 2/4] END C=1.0, kernel=linear, random_state=4;, score=0.841 total time=  38.0s
[CV 3/4] END C=1.0, kernel=linear, random_state=4;, score=0.836 total time=  38.2s
[CV 4/4] END C=1.0, kernel=linear, random_state=4;, score=0.833 total time=  38.4s
[CV 1/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.864 total time=  55.2s
[CV 2/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.868 total time=  55.4s
[CV 3/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.865 total time=  56.4s
[CV 4/4] END .C=1.0, kernel=rbf, random_state=4;, score=0.864 total time=  56.5s
[CV 1/4] END C=0.1, kernel=linear, random_state=4;, score=0.834 total time=  37.3s
[CV 2/4] END C=0.1, kernel=linear, random_state=4;, score=0.841 total time=  37.3s
[CV 3/4] END C=0.1, kernel=linear, random_state=4;, score=0.834 total time=  37.3s
[CV 4/4] END C=0.1, kernel=linear, random_state=4;, score=0.831 total time=  37.5s
[CV 1/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.821 total time= 1.4min
[CV 2/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.828 total time= 1.4min
[CV 3/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.824 total time= 1.4min
[CV 4/4] END .C=0.1, kernel=rbf, random_state=4;, score=0.821 total time= 1.4min
[CV 1/4] END C=0.01, kernel=linear, random_state=4;, score=0.817 total time=  49.2s
[CV 2/4] END C=0.01, kernel=linear, random_state=4;, score=0.823 total time=  49.4s
[CV 3/4] END C=0.01, kernel=linear, random_state=4;, score=0.819 total time=  49.3s
[CV 4/4] END C=0.01, kernel=linear, random_state=4;, score=0.816 total time=  49.7s
[CV 1/4] END C=0.01, kernel=rbf, random_state=4;, score=0.758 total time= 2.6min
[CV 2/4] END C=0.01, kernel=rbf, random_state=4;, score=0.761 total time= 2.5min
[CV 3/4] END C=0.01, kernel=rbf, random_state=4;, score=0.756 total time= 2.5min
[CV 4/4] END C=0.01, kernel=rbf, random_state=4;, score=0.756 total time= 2.5min

Tempo de execucao SVM:  3701 segundos ou 61.69 minutos
O melhor modelo e:  SVC(C=10.0, random_state=4)

-------------------
Execucao do teste 2
-------------------

Dividindo o conjunto de treinamento para a CV
---------------------------------------------
Numero de amostras para treinamento:  45000
Numero de amostras para validacao:  15000

Ajuste do modelo usando Regressao Logistica
-------------------------------------------
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[CV 1/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.826 total time=  51.0s
[CV 2/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.821 total time= 1.1min
[CV 3/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.829 total time=  27.8s
[CV 4/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.824 total time= 1.5min
[CV 5/5] END C=100.0, random_state=4, solver=newton-cg;, score=0.827 total time=  58.3s
[CV 1/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.8s
[CV 2/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.821 total time=   2.7s
[CV 3/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.7s
[CV 4/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.8s
[CV 5/5] END C=100.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.7s
[CV 1/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.827 total time=  32.2s
[CV 2/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.821 total time=  34.6s
[CV 3/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.831 total time=  42.9s
[CV 4/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.824 total time=  25.4s
[CV 5/5] END C=10.0, random_state=4, solver=newton-cg;, score=0.828 total time=  21.1s
[CV 1/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.8s
[CV 2/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.7s
[CV 3/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.826 total time=   2.8s
[CV 4/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.821 total time=   2.7s
[CV 5/5] END C=10.0, random_state=4, solver=lbfgs;, score=0.829 total time=   2.7s
[CV 1/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.827 total time=  16.1s
[CV 2/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.822 total time=  10.9s
[CV 3/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.830 total time=  10.9s
[CV 4/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.825 total time=  12.6s
[CV 5/5] END C=1.0, random_state=4, solver=newton-cg;, score=0.831 total time=  12.8s
[CV 1/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.824 total time=   2.7s
[CV 2/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.822 total time=   2.8s
[CV 3/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.826 total time=   2.8s
[CV 4/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.823 total time=   2.9s
[CV 5/5] END C=1.0, random_state=4, solver=lbfgs;, score=0.830 total time=   2.7s
[CV 1/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.823 total time=   8.5s
[CV 2/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.820 total time=   5.5s
[CV 3/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.830 total time=   6.5s
[CV 4/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.824 total time=   5.5s
[CV 5/5] END C=0.1, random_state=4, solver=newton-cg;, score=0.829 total time=   8.0s
[CV 1/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.822 total time=   2.7s
[CV 2/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.820 total time=   2.7s
[CV 3/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.830 total time=   2.8s
[CV 4/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.823 total time=   2.7s
[CV 5/5] END C=0.1, random_state=4, solver=lbfgs;, score=0.829 total time=   2.8s
[CV 1/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.802 total time=   4.4s
[CV 2/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.801 total time=   8.9s
[CV 3/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.807 total time=   6.6s
[CV 4/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.806 total time=   4.5s
[CV 5/5] END C=0.01, random_state=4, solver=newton-cg;, score=0.811 total time=   4.1s
[CV 1/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.802 total time=   2.8s
[CV 2/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.800 total time=   2.8s
[CV 3/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.807 total time=   2.7s
[CV 4/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.806 total time=   2.8s
[CV 5/5] END C=0.01, random_state=4, solver=lbfgs;, score=0.812 total time=   2.8s

Tempo de execucao Regressao Logistica:  660 segundos ou 11.0 minutos

Ajuste do modelo usando SVM
---------------------------
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[CV 1/5] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.3min
[CV 2/5] END C=100.0, kernel=linear, random_state=4;, score=0.828 total time= 5.3min
[CV 3/5] END C=100.0, kernel=linear, random_state=4;, score=0.834 total time= 5.5min
[CV 4/5] END C=100.0, kernel=linear, random_state=4;, score=0.833 total time= 5.2min
[CV 5/5] END C=100.0, kernel=linear, random_state=4;, score=0.833 total time= 5.5min
[CV 1/5] END C=100.0, kernel=rbf, random_state=4;, score=0.873 total time=  48.3s
[CV 2/5] END C=100.0, kernel=rbf, random_state=4;, score=0.873 total time=  48.4s
[CV 3/5] END C=100.0, kernel=rbf, random_state=4;, score=0.876 total time=  48.8s
[CV 4/5] END C=100.0, kernel=rbf, random_state=4;, score=0.876 total time=  49.1s
[CV 5/5] END C=100.0, kernel=rbf, random_state=4;, score=0.876 total time=  48.7s
[CV 1/5] END C=10.0, kernel=linear, random_state=4;, score=0.830 total time= 1.1min
[CV 2/5] END C=10.0, kernel=linear, random_state=4;, score=0.831 total time= 1.1min
[CV 3/5] END C=10.0, kernel=linear, random_state=4;, score=0.835 total time= 1.1min
[CV 4/5] END C=10.0, kernel=linear, random_state=4;, score=0.834 total time= 1.1min
[CV 5/5] END C=10.0, kernel=linear, random_state=4;, score=0.835 total time= 1.1min
[CV 1/5] END C=10.0, kernel=rbf, random_state=4;, score=0.878 total time=  44.1s
[CV 2/5] END C=10.0, kernel=rbf, random_state=4;, score=0.879 total time=  44.2s
[CV 3/5] END C=10.0, kernel=rbf, random_state=4;, score=0.881 total time=  44.3s
[CV 4/5] END C=10.0, kernel=rbf, random_state=4;, score=0.885 total time=  44.5s
[CV 5/5] END C=10.0, kernel=rbf, random_state=4;, score=0.881 total time=  44.5s
[CV 1/5] END C=1.0, kernel=linear, random_state=4;, score=0.833 total time=  35.2s
[CV 2/5] END C=1.0, kernel=linear, random_state=4;, score=0.832 total time=  35.1s
[CV 3/5] END C=1.0, kernel=linear, random_state=4;, score=0.837 total time=  35.5s
[CV 4/5] END C=1.0, kernel=linear, random_state=4;, score=0.835 total time=  35.4s
[CV 5/5] END C=1.0, kernel=linear, random_state=4;, score=0.838 total time=  35.3s
[CV 1/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.862 total time=  49.1s
[CV 2/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.862 total time=  49.1s
[CV 3/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.865 total time=  49.2s
[CV 4/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.865 total time=  49.2s
[CV 5/5] END .C=1.0, kernel=rbf, random_state=4;, score=0.870 total time=  49.7s
[CV 1/5] END C=0.1, kernel=linear, random_state=4;, score=0.834 total time=  33.9s
[CV 2/5] END C=0.1, kernel=linear, random_state=4;, score=0.832 total time=  33.8s
[CV 3/5] END C=0.1, kernel=linear, random_state=4;, score=0.838 total time=  34.1s
[CV 4/5] END C=0.1, kernel=linear, random_state=4;, score=0.833 total time=  34.0s
[CV 5/5] END C=0.1, kernel=linear, random_state=4;, score=0.837 total time=  34.0s
[CV 1/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.823 total time= 1.2min
[CV 2/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.817 total time= 1.2min
[CV 3/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.822 total time= 1.2min
[CV 4/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.819 total time= 1.2min
[CV 5/5] END .C=0.1, kernel=rbf, random_state=4;, score=0.829 total time= 1.2min
[CV 1/5] END C=0.01, kernel=linear, random_state=4;, score=0.813 total time=  45.3s
[CV 2/5] END C=0.01, kernel=linear, random_state=4;, score=0.817 total time=  45.1s
[CV 3/5] END C=0.01, kernel=linear, random_state=4;, score=0.822 total time=  45.4s
[CV 4/5] END C=0.01, kernel=linear, random_state=4;, score=0.813 total time=  45.2s
[CV 5/5] END C=0.01, kernel=linear, random_state=4;, score=0.825 total time=  45.3s
[CV 1/5] END C=0.01, kernel=rbf, random_state=4;, score=0.755 total time= 2.3min
[CV 2/5] END C=0.01, kernel=rbf, random_state=4;, score=0.752 total time= 2.3min
[CV 3/5] END C=0.01, kernel=rbf, random_state=4;, score=0.758 total time= 2.3min
[CV 4/5] END C=0.01, kernel=rbf, random_state=4;, score=0.756 total time= 2.3min
[CV 5/5] END C=0.01, kernel=rbf, random_state=4;, score=0.764 total time= 2.3min

Tempo de execucao SVM:  4328 segundos ou 72.15 minutos
O melhor modelo e:  SVC(C=10.0, random_state=4)
Tempo total:  167.74  minutos ou 2.8  horas.
