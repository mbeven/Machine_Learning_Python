
#################
# set up script #
#################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('/Users/michaelbeven/documents/09_Phoenix_Analysis/03 Python/02 Machine Learning/Packt/C2')
import Perceptron

########
# data #
########

df = pd.read_csv('../Data/iris/iris.data.txt',header=None)

y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values

########
# plot #
########

plt.scatter(X[:50,0],X[:50,1],
        color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],
        color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

############
# analysis #
############

ppn = Perceptron.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)

########
# plot #
########

plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()


