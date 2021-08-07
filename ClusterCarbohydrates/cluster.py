# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 09:57:58 2020

@author: glen2
"""

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
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# Reading the patient insulin and csv file
ins1 = pd.read_csv(r'InsulinData1.csv',index_col = 'Index').iloc[::-1].reset_index(drop = True)
cgm1 = pd.read_csv(r'CGMData1.csv', index_col = "Index").iloc[::-1].reset_index(drop = True)

# creating datetime col
ins1['DateTime'] = ins1['Date'] + ' '+ ins1['Time']
ins1['DateTime'] = pd.to_datetime(ins1['DateTime'], format= ('%m/%d/%Y %H:%M:%S'))
cgm1['DateTime'] = cgm1['Date'] + ' '+ cgm1['Time']
cgm1['DateTime'] = pd.to_datetime(cgm1['DateTime'], format= ('%m/%d/%Y %H:%M:%S'))


# extracting meal data and ground truth
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
            meal_data.append(((mealdf['DateTime'][i]- pd.Timedelta('30 minutes')), x))   
            
        else:
            continue
            
    
    # Store meal data in N X 30 CSV

    count = 0
    data=[]
    carb_vals = []
    for start, end in meal_data:
        time = start+pd.Timedelta('30 minutes')
        y = cgmdf[(cgmdf['DateTime'] > start) & (cgmdf['DateTime'] <= end)]
        if len(y) == 30:
            if y['Sensor Glucose (mg/dL)'].isnull().values.any():
                continue
            data.append(y['Sensor Glucose (mg/dL)'].values)
            c = mealdf[mealdf['DateTime'] == time]
            carb_vals.append(float(c['BWZ Carb Input (grams)'].values))


    # ground truth vector for meals
    ground_t = []
    # no_of_bins = max_carb_val - min_carb_val
    
    for i in range(len(carb_vals)):
        if 3.0 <= carb_vals[i] < 23.0:
            ground_t.append(0)
        elif 23.0 <= carb_vals[i] < 43.0:
            ground_t.append(1)
        elif 43.0 <= carb_vals[i] < 63.0:
            ground_t.append(2)
        elif 63.0 <= carb_vals[i] < 83.0:
            ground_t.append(3)
        elif 83.0 <= carb_vals[i] < 103.0:
            ground_t.append(4)
        elif 103.0 <= carb_vals[i] < 129.0:
            ground_t.append(5)

    ground_truth = pd.DataFrame(ground_t)
                
            
    meal = pd.DataFrame(data)
    
    return(meal, ground_truth) 

# extracting features from meal data    
def featureextract(meal):
    
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
        k = x.index(max(x)) - 6
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
    

    ## meal feature vector##

    mealfv = pd.concat((feat1m,feat2m,feat34m,feat56789m),axis=1)

    return mealfv



def main():
    
    Meal,GroundTruth = MealNomeal(ins1,cgm1)
    
    Featurevec = featureextract(Meal)
    
    # scaling and normalizing feature vector
    sc = StandardScaler()
    Featurevec = sc.fit_transform(Featurevec)
    Featurevec = normalize(Featurevec)
    
    
    # Clustering (KMeans)
    clusters= KMeans(n_clusters = 6, random_state =0).fit(Featurevec)
    x = clusters.cluster_centers_

    #sse for k- means
    sse=[]
    for i in range(len(Featurevec)):
        if  clusters.labels_[i]== 0:
            sse.append((Featurevec[i] - x[0])**2)
        elif clusters.labels_[i] == 1:
            sse.append((Featurevec[i] - x[1])**2)
        elif clusters.labels_[i] == 2:
            sse.append((Featurevec[i] - x[2])**2)
        elif clusters.labels_[i] == 3:
            sse.append((Featurevec[i] - x[3])**2)
        elif clusters.labels_[i] == 4:
            sse.append((Featurevec[i] - x[4])**2)
        elif clusters.labels_[i] == 5:
            sse.append((Featurevec[i] - x[5])**2)
    y = sum(sse)
    sse_kmeans = y.sum()
    
    # entropy and purity  for kmeans
    
    
    # creating matrix for entropy and purity calculation
    kmeans_labels = pd.DataFrame(clusters.labels_.astype(int), columns=['KMeans'])
    labels = pd.concat((GroundTruth, kmeans_labels), axis= 1)
    
    c1 = labels[labels['KMeans']==0]
    b1=c1[0].value_counts().to_frame().sort_index().transpose()
    
    c2 = labels[labels['KMeans']==1]
    b2=c2[0].value_counts().to_frame().sort_index().transpose()
    
    c3 = labels[labels['KMeans']==2]
    b3 = c3[0].value_counts().to_frame().sort_index().transpose()
    
    c4 = labels[labels['KMeans']==3]
    b4 = c4[0].value_counts().to_frame().sort_index().transpose()
    
    c5 = labels[labels['KMeans']==4]
    b5 = c5[0].value_counts().to_frame().sort_index().transpose()
    
    c6 = labels[labels['KMeans']==5]
    b6 = c6[0].value_counts().to_frame().sort_index().transpose()
    
    matrix = pd.concat((b1,b2,b3,b4,b5,b6), axis =0).fillna(0)
    
    
    l = matrix.to_numpy(dtype = int)

    # calculating entropy and purity
    whole = sum(l).sum()
    entropy_w = 0
    purity_w = 0
    entropy = []
    purity = []
    for i in range(len(l)):
        tot = sum(l[i])
        e = 0
        p = 0
        for j in l[i]:
            if j==0:
                continue
            e += (-j/tot)*np.log2(j/tot)
            p = (max(l[i])/tot)
        entropy.append(e)
        purity.append(p)
        
        entropy_w += ((tot/whole)*entropy[i])
        purity_w += ((tot/whole)*purity[i])
    
    Entropy_Kmeans = entropy_w
    Purity_Kmeans = purity_w    
    
    # dbscan clustering

    
    db = DBSCAN(eps=1.57, min_samples=1).fit_predict(Featurevec)
    
    # we obtain 0 as the only cluster after running dbscan
    
    # therefore running k means to split the cluster into 6 clusters
    
    clusters2= KMeans(n_clusters = 6, random_state = 10).fit(Featurevec)
    x2 = clusters2.cluster_centers_
    #sse for dbscan
    sse2=[]
    for i in range(len(Featurevec)):
        if  clusters2.labels_[i]== 0:
            sse2.append((Featurevec[i] - x2[0])**2)
        elif clusters2.labels_[i] == 1:
            sse2.append((Featurevec[i] - x2[1])**2)
        elif clusters2.labels_[i] == 2:
            sse2.append((Featurevec[i] - x2[2])**2)
        elif clusters2.labels_[i] == 3:
            sse2.append((Featurevec[i] - x2[3])**2)
        elif clusters2.labels_[i] == 4:
            sse2.append((Featurevec[i] - x2[4])**2)
        elif clusters2.labels_[i] == 5:
            sse2.append((Featurevec[i] - x2[5])**2)
    y2 = sum(sse2)
    sse_dbscan = y2.sum()
    
    # entropy and purity  for dbscan

    kmeans_labels = pd.DataFrame(clusters2.labels_.astype(int), columns=['KMeans'])
    labels = pd.concat((GroundTruth, kmeans_labels), axis= 1)
    
    c1 = labels[labels['KMeans']==0]
    b1=c1[0].value_counts().to_frame().sort_index().transpose()
    
    c2 = labels[labels['KMeans']==1]
    b2=c2[0].value_counts().to_frame().sort_index().transpose()
    
    c3 = labels[labels['KMeans']==2]
    b3 = c3[0].value_counts().to_frame().sort_index().transpose()
    
    c4 = labels[labels['KMeans']==3]
    b4 = c4[0].value_counts().to_frame().sort_index().transpose()
    
    c5 = labels[labels['KMeans']==4]
    b5 = c5[0].value_counts().to_frame().sort_index().transpose()
    
    c6 = labels[labels['KMeans']==5]
    b6 = c6[0].value_counts().to_frame().sort_index().transpose()
    
    matrix = pd.concat((b1,b2,b3,b4,b5,b6), axis =0).fillna(0)
    
    
    l =matrix.to_numpy(dtype = int)
    
    
    whole = sum(l).sum()
    entropy_w = 0
    purity_w = 0
    entropy = []
    purity = []
    for i in range(len(l)):
        tot = sum(l[i])
        e = 0
        p = 0
        for j in l[i]:
            if j==0:
                continue
            e += (-j/tot)*np.log2(j/tot)
            p = (max(l[i])/tot)
        entropy.append(e)
        purity.append(p)
        
        entropy_w += ((tot/whole)*entropy[i])
        purity_w += ((tot/whole)*purity[i])
    
    Entropy_dbscan = entropy_w
    Purity_dbsdcan = purity_w
    
    # storing results
    
    data = {'SSE for Kmeans': [sse_kmeans], 'SSE for DBSCAN':[sse_dbscan],'Entropy for Kmeans': [Entropy_Kmeans], 'Entropy for DBSCAN':[Entropy_dbscan], 'Purity for K means':[Purity_Kmeans], 'Purity for DBSCAN': [Purity_dbsdcan]}

    results = pd.DataFrame(data)
    results.to_csv(r'Results.csv')
    
    
    
if __name__ == "__main__":
    main()
    
    
    


