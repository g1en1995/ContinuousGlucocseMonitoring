# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:42:59 2020

@author: glen2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle



with open(r'MealNomealmodel.pkl','rb') as f:
    svm_model = pickle.load(f)
    
test = pd.read_csv(r'test.csv')

def featureextract(df):
    
    # FEATURE EXTRACTION

    val1m=[]
    val2m=[]
    val3m=[]
    val4m=[]
    val5m=[]
    val6m=[]
    val7m=[]
    val8m=[]
    val9m=[]
    val10m=[]
    val11m=[]
    
    for i in range(len(df)):
        # feature1 # # Meal absorption time
        
        x = list(df.loc[i])
        k = x.index(max(x)) - 0
        val1m.append(k*5)
    
        #feature2# #cgmMax - cgmmin
        
        val2m.append(max(df.loc[i])-(min(df.loc[i])))
        
        #feature34#  #Peak velocity and time of peak velocity 
        
        vel = list(np.gradient(df.loc[i]))
        val3m.append(max(vel))
        val4m.append(vel.index(max(vel)))
        
        #feature56789# rolling means
        
        x = df.loc[i].rolling(window = 3).mean()
        val5m.append(x[8])
        val6m.append(x[9])
        val7m.append(x[10])
        val8m.append(x[11])
        val9m.append(x[12])
        
        #feature1011# fft
        
        j = df.loc[i]
        x = list(np.fft.fft(j))
        y = list(np.fft.fft(j))
        x.sort()
        val10m.append(y.index(x[1]))
        val11m.append(y.index(x[2]))

      
    data1m = {"Feature1": val1m}
    feat1m = pd.DataFrame(data1m, columns=['Feature1'])
    
    data2m = {"Feature2": val2m}
    feat2m = pd.DataFrame(data2m, columns=['Feature2'])
    
    data34m = {"Feature3": val3m, "Feature4": val4m}
    feat34m = pd.DataFrame(data34m, columns=['Feature3', 'Feature4'])
    
    data56789m = {'Feature5' : val5m, 'Feature6' : val6m, 'Feature7': val7m, 'Feature8':val8m, 'Feature9': val9m}
    feat56789m = pd.DataFrame(data56789m, columns=['Feature5','Feature6','Feature7','Feature8','Feature9'])
    
    data1011m = {"Feature10": val10m, "Feature11": val11m}
    feat1011m = pd.DataFrame(data1011m, columns=['Feature10', 'Feature11'])
    
    fv = pd.concat((feat1m,feat2m,feat34m,feat56789m),axis=1)
    
    
    return fv

fv = featureextract(test)
fv['Feature1'] = fv['Feature1'].astype('float')
fv['Feature4'] = fv['Feature4'].astype('float')
print(fv.info())

sc = StandardScaler()

X_test_s = sc.fit_transform(fv)

y_pred = svm_model.predict(X_test_s)

y = pd.DataFrame(y_pred)

y.to_csv('Result.csv')

    
