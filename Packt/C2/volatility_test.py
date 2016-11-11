# Michael Beven
# 20161110
# Volatility Test Using Chapter 2 Materials

#########
# setup #
#########

import pandas as pd
from pandas.io.data import DataReader
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os
os.chdir('/Users/michaelbeven/documents/09_Phoenix_Analysis/03 Python/02 Machine Learning/Packt/C2')
import Perceptron
import AdalineGD
import AdalineSGD

start_date = datetime(2005,1,1)
end_date = time.strftime('%x') #today
index = '^SP500TR'

########
# data #
########

df = pd.io.data.DataReader(index, 'yahoo', start_date, end_date)
df['logrets_squared'] = (np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1)))**2

y = df.iloc[:,6].values
y = np.where(y >= 0.00005, 1, -1)
X = df.iloc[:,[1,2]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

############
# analysis #
############

ppn = Perceptron.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)

ada = AdalineGD.AdalineGD(eta=0.01, n_iter=15)
ada.fit(X_std, y)

adaS = AdalineSGD.AdalineSGD(eta=0.01, n_iter=15, random_state=1)
adaS.fit(X_std,y)

########
# plot #
########

plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

plt.plot(range(1,len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

plt.plot(range(1,len(adaS.cost_) + 1), adaS.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
