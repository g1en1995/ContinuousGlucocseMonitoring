# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:38:52 2020

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

ins1 = pd.read_csv(r'InsulinData1.csv',index_col = 'Index').iloc[::-1].reset_index(drop = True)
ins2 = pd.read_csv(r'InsulinData2.csv', index_col = 'Index').iloc[::-1].reset_index(drop = True)
cgm1 = pd.read_csv(r'CGMData1.csv', index_col = "Index").iloc[::-1].reset_index(drop = True)
cgm2 = pd.read_csv(r'CGMData2.csv',index_col = "Index").iloc[::-1].reset_index(drop = True)

ins1['DateTime'] = ins1['Date'] + ' '+ ins1['Time']
ins1['DateTime'] = pd.to_datetime(ins1['DateTime'], format= ('%m/%d/%Y %H:%M:%S'))
cgm1['DateTime'] = cgm1['Date'] + ' '+ cgm1['Time']
cgm1['DateTime'] = pd.to_datetime(cgm1['DateTime'], format= ('%m/%d/%Y %H:%M:%S'))
ins2['DateTime'] = ins2['Date'] + ' '+ ins2['Time']
ins2['DateTime'] = pd.to_datetime(ins2['DateTime'], format= ('%d-%m-%Y %H:%M:%S'))
cgm2['DateTime'] = cgm2['Date'] + ' '+ cgm2['Time']
cgm2['DateTime'] = pd.to_datetime(cgm2['DateTime'], format= ('%d-%m-%Y %H:%M:%S'))

def MealNomeal(insdf, cgmdf):
    
    mealdf = insdf[insdf['BWZ Carb Input (grams)'].notnull()].reset_index(drop = True)
    mealdf = mealdf.loc[mealdf['BWZ Carb Input (grams)'] != 0].reset_index(drop = True)

    meal_data = []
    for i in range(len(mealdf)):
        x = mealdf['DateTime'][i] + pd.Timedelta('2 hours')
        y = insdf[ (insdf['DateTime'] <= x) & (insdf['DateTime'] >= mealdf['DateTime'][i])] 
        y = y[y['BWZ Carb Input (grams)'].notnull()].reset_index()
        y = y[y['BWZ Carb Input (grams)'] != 0].reset_index()
        if len(y) == 1:
            meal_data.append(((mealdf['DateTime'][i] - pd.Timedelta('30 minutes')), x))
        else:
            continue

    #store meal data and no meal data in N X 30 and N X 24 CSV

    count = 0
    data=[]
    for start, end in meal_data:
        y = cgmdf[(cgmdf['DateTime'] > start) & (cgmdf['DateTime'] <= end)]
        if len(y) == 30:
            if y['Sensor Glucose (mg/dL)'].isnull().values.any():
                continue
            #l =  list(y['Sensor Glucose (mg/dL)'].values)
            #if l.index(max(l)) < 8 or l.index(max(l)) > 26:
            #    continue
            #if 10 < l.index(min(l)) < 15:
            #    continue

            data.append(y['Sensor Glucose (mg/dL)'].values)
            count += 1
    meal = pd.DataFrame(data)
    
    #nomeal data
    
    nomeal_data = []
    final = cgmdf.iloc[-1]['DateTime'] - pd.Timedelta('2 hours')
    x = cgmdf.iloc[0]['DateTime']
    while x < final:
        k = x + pd.Timedelta('2 hours')

        y = insdf[(insdf['DateTime'] > x)  & (insdf['DateTime'] <= k)]
        y = y[y['BWZ Carb Input (grams)'].notnull()].reset_index()
        y = y[y['BWZ Carb Input (grams)'] != 0].reset_index()
        if len(y)>0:
            x = y['DateTime'][len(y)-1] + pd.Timedelta('2 hours')
        elif len(y) == 0:
            nomeal_data.append((x, x + pd.Timedelta('2 hours')))
            x = x + pd.Timedelta('2 hours')

    count = 0
    data=[]
    for start, end in nomeal_data:
        y = cgmdf[(cgmdf['DateTime'] > start) & (cgmdf['DateTime'] <= end)]
        if len(y) == 24:
            if count == 400:
                break
            if y['Sensor Glucose (mg/dL)'].isnull().values.any():
                continue
            data.append(y['Sensor Glucose (mg/dL)'].values)
            count += 1

    nomeal = pd.DataFrame(data)
    
    return(meal, nomeal) 
    
    
def featureextract(meal,nomeal=None):
    
    # MEAL FEATURE EXTRACTION
   
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
    for i in range(len(meal)):
        # feature1 # # Meal absorption time
        
        x = list(meal.loc[i])
        k = x.index(max(x)) - 5
        val1m.append(k*5)
    
        #feature2# #cgmMax - cgmmin
        
        val2m.append(max(meal.loc[i])-(min(meal.loc[i])))
        
        #feature34#  #Peak velocity and time of peak velocity 
        
        vel = list(np.gradient(meal.loc[i]))
        val3m.append(max(vel))
        val4m.append(vel.index(max(vel)))
        
        #feature56789# rolling means
        
        x = meal.loc[i].rolling(window = 3).mean()
        val5m.append(x[8])
        val6m.append(x[9])
        val7m.append(x[10])
        val8m.append(x[11])
        val9m.append(x[12])
        
        #feature1011# fft
        
        j = meal.loc[i]
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
    

    # NOMEAL FEATURE EXTRACTION
    val1nm=[]
    val2nm=[]
    val3nm=[]
    val4nm=[]
    val5nm=[]
    val6nm=[]
    val7nm=[]
    val8nm=[]
    val9nm=[]
    val10nm=[]
    val11nm=[]
    for i in range(len(nomeal)):
        
        # feature1 # 
        x = list(nomeal.loc[i])
        k = x.index(max(x)) - 0
        val1nm.append(k*5)
    
        #feature2#
        val2nm.append(max(nomeal.loc[i])-(min(nomeal.loc[i])))
        
        #feature34#
        vel = list(np.gradient(nomeal.loc[i]))
        val3nm.append(max(vel))
        val4nm.append(vel.index(max(vel)))
        
        #feature56789# rolling means
        
        x = nomeal.loc[i].rolling(window = 3).mean()
        val5nm.append(x[8])
        val6nm.append(x[9])
        val7nm.append(x[10])
        val8nm.append(x[11])
        val9nm.append(x[12])   
        
        #feature1011# fft
        
        j = nomeal.loc[i]
        x = list(np.fft.fft(j))
        y = list(np.fft.fft(j))
        x.sort()
        val10nm.append(y.index(x[1]))
        val11nm.append(y.index(x[2]))
        
    data1nm = {"Feature1": val1nm}
    feat1nm = pd.DataFrame(data1nm, columns=['Feature1'])
        
    data2nm = {"Feature2": val2nm}
    feat2nm = pd.DataFrame(data2nm, columns=['Feature2'])
      
    data34nm = {"Feature3": val3nm, "Feature4": val4nm}
    feat34nm = pd.DataFrame(data34nm, columns=['Feature3', 'Feature4'])
    
    data56789nm = {'Feature5' : val5nm, 'Feature6' : val6nm, 'Feature7': val7nm, 'Feature8':val8nm, 'Feature9': val9nm}
    feat56789nm = pd.DataFrame(data56789nm, columns=['Feature5','Feature6','Feature7','Feature8','Feature9'])
    
    data1011nm = {"Feature10": val10nm, "Feature11": val11nm}
    feat1011nm = pd.DataFrame(data1011nm, columns=['Feature10', 'Feature11'])


    ## meal feature vector##

    classvar = {'class' : np.ones(len(meal),dtype=int)}
    y = pd.DataFrame(classvar, columns=['class'])
    mealfv = pd.concat((feat1m,feat2m,feat34m,feat56789m, y),axis=1)

    ## nomeal features vector ##
    classvar = {'class' : np.zeros(len(nomeal),dtype=int)}
    y = pd.DataFrame(classvar, columns=['class'])
    nomealfv = pd.concat((feat1nm,feat2nm, feat34nm,feat56789nm, y),axis=1)
    
    featurevec = pd.concat((mealfv, nomealfv), axis=0).reset_index(drop=True)
    
    return featurevec

def predict(featurevec):
    X = featurevec.iloc[:,:-1].values
    y = featurevec.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=  0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_s = sc.transform(X_train)
    X_test_s = sc.transform(X_test)
    svm_model = SVC(kernel='rbf', C = 1.0, random_state = 3)
    svm_model.fit(X_train_s, y_train)
    y_pred = svm_model.predict(X_test_s)
    filename = r'MealNomealmodel.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(svm_model, f)
    
    print('\nSupport Vector Machine Analysis\n')
    print('Number in test', len(y_test))
    print('Misclassified samples: %d' %(y_test !=  y_pred).sum())
    print('Accuracy: %.3f' %accuracy_score(y_test, y_pred))

        
meal1,nomeal1 = MealNomeal(ins1,cgm1)
meal2,nomeal2 = MealNomeal(ins2,cgm2)

meal = pd.concat((meal1,meal2), axis =0).reset_index(drop = True)
nomeal = pd.concat((nomeal1,nomeal2), axis = 0).reset_index(drop = True)

featurevec = featureextract(meal,nomeal)

predict(featurevec)