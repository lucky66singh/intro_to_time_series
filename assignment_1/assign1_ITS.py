# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:34:57 2019

@author: reino
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
import numpy as np

def main():
#Opdracht 1
    print("OPDRACHT 1")
    data = pd.read_csv('data_assign_p1.csv')
    titels = list(data)
    #print(titels)
    #print (data['obs'][0])
    X = data [['obs']]
    X1 = []
    Y = data [['GDP_QGR']]
    Z = []
    for i in range (0,88):
        X1.append(i)
        Z.append(data['obs'][i])
    
    plt.plot(Z, Y, linewidth=2.0)
    plt.xlabel('date (1987Q2 - 2009Q1)')
    plt.ylabel('GDP growth rate') #ten opzichte van jaar ervoor
    plot_acf(Y, lags=4) #x = delay/tijdstip
    plot_pacf(Y, lags=4)
    #plot_acf(X1)
    plt.show()
    print("\n")
    print("OPDRACHT 2") 
    pd.plotting.lag_plot(Y) #correlatie y(t) en y(t+1)
    #decomposed = seasonal_decompose(Y, model = 'additive')
    #decomposed.plot()
    model = ARMA(Y, order=(1,1))
    model_fit = model.fit()
    print(model_fit.summary())
    
    #data = data.join(data)
    model2 = AR(Y)
    model2_fit = model2.fit(3)
    #print(model_fit.summary())
    print('The lag value chose is: %s' % model2_fit.k_ar)
    print('The coefficients of the model are:\n %s' % model2_fit.params)
    print(model.loglike(model_fit.params))
    ypred = model2_fit.predict(Y, start = Y.index[3])
    print(ypred)
    
   
    
    
if __name__ == "__main__":
    main()

