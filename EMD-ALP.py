########################### IMPORTING ########################################

import numpy as np
import pandas as pd
import sys
import scipy.io
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
import warnings
import itertools
import seaborn as sns

if __name__ == "__main__":
    from PyEMD import EMD
from scipy.signal import hilbert
from PyEMD.compact import filt6, pade6

from statsmodels.tsa.stattools import adfuller

######################## LOADING THE DATASET #################################
df = pd.read_csv("C:/Users/משתמש/Desktop/פרוייקט/DataSets/train.csv")
print(df)

columns = df.columns.tolist() 
columns_lst = []
for col in columns:
    data_col=df[col].to_frame()
    columns_lst.append(data_col)
   
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

######################### Iterative KNN #######################################  

for i in range(1,99):
    
    warnings.filterwarnings('ignore')
    
    data = columns_lst[i].iloc[4000:4600].reset_index(drop=True)
    amplitude = np.array(data.values.flatten().tolist())
    Location_amplitude = rolling_window(amplitude,8)
    
    intervals_lst = []  
    for j in range(len(Location_amplitude)):
        extrected_values = Location_amplitude [j].flatten().tolist()
        intervals_lst.append(extrected_values)
        
    Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()
     
    #Splitting the data into train and test sets
    train_location = Location_amplitude.iloc[:300] 
    test_location = Location_amplitude.iloc[300:]
    
    train_x_location = train_location.iloc[:,:-1] #data
    train_y_location = train_location.iloc[:,-1] #labels
    test_x_location = test_location.iloc[:,:-1]
    test_y_location = test_location.iloc[:,-1]
    
    #find the best k value
    KNN_para = {'n_neighbors': [4,5,6,7,8,9],
                'algorithm' : ["auto",'brute'], 
                'weights': ['uniform','distance']}
    
    knn = neighbors.KNeighborsRegressor()
    
    knnGS = GridSearchCV(knn,KNN_para)
    knnGS.fit(train_x_location,train_y_location)
    #print ("\n"+'K-neighbors chosen parameters (EMD): {}'.format(knnGS.best_params_))
    
    # implementing the best parameters to model.
    knn_clf = neighbors.KNeighborsRegressor(**knnGS.best_params_)
    
    knn_clf.fit(train_x_location,train_y_location)
    
    predicted_values_location= []
    for sample in range(len(test_x_location.values)):
        train_x_location = train_location.iloc[:,:-1] #data 
        train_y_location = train_location.iloc[:,-1] #labels
        test_x_location = test_location.iloc[:,:-1]
        knn_clf = neighbors.KNeighborsRegressor(**knnGS.best_params_)
        knn_clf.fit(train_x_location,train_y_location)
        pred_y=knn_clf.predict(test_x_location.values[sample].reshape(1,-1))
        predicted_values_location.append(pred_y.item())
        update_sample = np.append(test_x_location.values[sample], pred_y).tolist()
        train_location.loc[train_location.index.max() + 1, :] = update_sample
    
    real = test_y_location.reset_index(drop=True)
        
    mse = mean_squared_error(real,predicted_values_location)
    rmse = sqrt(mse)
    print(rmse)
    #print('RMSE for the iterative version of KNN using time series '+str(i)+' : %.3f' % rmse)
  
############################### EMD AND ITERATIVE KNN ########################
    
import sys
sys.path.append("C:/Users/משתמש/Desktop/פרוייקט/Code")
from EMD_Visualisation import Visualisation    

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

for i in range(1,99):

    data = columns_lst[i].iloc[4000:4600].reset_index(drop=True)
    
    amplitude = np.array(data.values.flatten().tolist())

    # Define signal
    t = np.arange(0,600)
    s = amplitude
    
    emd = EMD()
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    # _e a plot with all IMFs and residue
    #vis = Visualisation(emd)
    #vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    
    imfs_without_first = [sum(i) for i in zip(*imfs[1:])]
    
    Series = res + imfs_without_first
    
    Location_amplitude = rolling_window(Series,8)
        
    intervals_lst = []  
    for j in range(len(Location_amplitude)):
        extrected_values = Location_amplitude [j].flatten().tolist()
        intervals_lst.append(extrected_values)
            
    Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()
        
    #Splitting the data into train and test sets
    train_location = Location_amplitude.iloc[:300] 
    test_location = Location_amplitude.iloc[300:]
        
    train_x_location = train_location.iloc[:,:-1] #data
    train_y_location = train_location.iloc[:,-1] #labels
    test_x_location = test_location.iloc[:,:-1]
    test_y_location = test_location.iloc[:,-1]
        
    #find the best k value
    KNN_para = {'n_neighbors': [4,5,6,7,8,9],
                'algorithm' : ["auto",'brute'], 
                'weights': ['uniform','distance']}
        
    knn = neighbors.KNeighborsRegressor()
        
    knnGS = GridSearchCV(knn,KNN_para)
    knnGS.fit(train_x_location,train_y_location)
    #print ("\n"+'K-neighbors chosen parameters (EMD): {}'.format(knnGS.best_params_))
        
    # implementing the best parameters to model.
    knn_clf = neighbors.KNeighborsRegressor(**knnGS.best_params_)
        
    knn_clf.fit(train_x_location,train_y_location)
        
    predicted_values_location= []
    for sample in range(len(test_x_location.values)):
        train_x_location = train_location.iloc[:,:-1] #data 
        train_y_location = train_location.iloc[:,-1] #labels
        test_x_location = test_location.iloc[:,:-1]
        knn_clf = neighbors.KNeighborsRegressor(**knnGS.best_params_)
        knn_clf.fit(train_x_location,train_y_location)
        pred_y=knn_clf.predict(test_x_location.values[sample].reshape(1,-1))
        predicted_values_location.append(pred_y.item())
        update_sample = np.append(test_x_location.values[sample], pred_y).tolist()
        train_location.loc[train_location.index.max() + 1, :] = update_sample
        
    real = test_y_location.reset_index(drop=True)
            
    mse = mean_squared_error(real,predicted_values_location)
    rmse = sqrt(mse)
    print(rmse)
    #print ('RMSE using EMD and iterative version of KNN for time series ' +str(i)+' : %.3f' % rmse)

############################### LAPLACIAN PYRAMID #############################

import ckwrap
import sys
sys.path.append("C:/Users/משתמש/Desktop/פרוייקט/Code")
from Laplacian_Pyramids import LP_Train, LP,calcEpsilon
epsilon_factor = 4

for i in range(1,99):
    
    data = columns_lst[i].iloc[4000:4600].reset_index(drop=True)
    
    amplitude = np.array(data.values.flatten().tolist())
    
    epsilon = calcEpsilon(range(len(amplitude)),amplitude)
    
    Location_amplitude = rolling_window(amplitude,8)    
    intervals_lst = []  
    for j in range(len(Location_amplitude)):
        extrected_values = Location_amplitude [j].flatten().tolist()
        intervals_lst.append(extrected_values)
       
    Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()
    
    #print(Location_amplitude)
        
    predicted_values = LP(Location_amplitude.iloc[:300],Location_amplitude.iloc[300:],epsilon,True)
    real = Location_amplitude.iloc[300:].iloc[:,-1].values  
    
    mse = mean_squared_error(real,predicted_values)
    rmse = sqrt(mse)
    print(rmse)
    #print ('RMSE using LP for time series ' +str(i)+' : %.3f' % rmse)

############################### EMD WITH LAPLACIAN PYRAMID ####################

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Finding the IMFs and res of a time-series.
def find_IMFs_and_res(dataframe):
    amplitude = np.array(dataframe.values.flatten().tolist())
    emd = EMD()
    emd.emd(amplitude)
    imfs, res = emd.get_imfs_and_residue()
    return imfs,res
    
def normalization(array):
    min_max_scaler = MinMaxScaler()
    array = array.reshape(-1, 1)
    array_scaled = min_max_scaler.fit_transform(array).flatten().tolist()
    return array_scaled

def chunks(lst,chunk):
    chunks_lst = [lst[x:x+chunk] for x in range(0, len(lst), chunk)]
    return chunks_lst

def find_imfs_median(imfs,chunk_size):
    median = []
    for imf in range(len(imfs)):
        normalize_imf = normalization(imfs[imf])
        chunk_imf = chunks(normalize_imf,chunk_size)
        std = []
        for j in range(len(chunk_imf)):
            std.append(np.std(chunk_imf[j]))
        median.append(statistics.median(std))
    return median

def find_clusters(lst,num_of_clusters,show_number):
    if show_number == True:
        km = ckwrap.ckmeans(lst,num_of_clusters)
        buckets = [ [] for _ in range(num_of_clusters)]
        for i in range(len(lst)):
            buckets[km.labels[i]].append((lst[i],i))
        return buckets
    elif show_number == False:
        km = ckwrap.ckmeans(lst,num_of_clusters)
        buckets = [ [] for _ in range(num_of_clusters)]
        for i in range(len(lst)):
            buckets[km.labels[i]].append(int(i))
        return buckets
    
def create_frame(series):
    amplitude = np.array(series.values.flatten().tolist())
    Location_amplitude = rolling_window(amplitude,8)
    intervals_lst = []  
    for j in range(len(Location_amplitude)):
        extrected_values = Location_amplitude [j].flatten().tolist()
        intervals_lst.append(extrected_values)       
    Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()
    return Location_amplitude   
    
def ALP_EMD(imfs,chunk_size,clusters,test_data,is_val):
    #finding the median
    median_lst = find_imfs_median(imfs,chunk_size)
    #creating clusters
    clusters_lst = find_clusters(median_lst,clusters,False)
    #The Series list contains two series consisting of the IMFs in the same cluster.
    series = [ [] for _ in range(len(clusters_lst))]
    for i in range(len(clusters_lst)):
        for j in range(len(clusters_lst[i])):
            imfs_idx = clusters_lst[i][j]
            series[i].append(imfs[imfs_idx])
        series[i] = [sum(j) for j in zip(*series[i])]
    #Using ALP to predict each cluster seperatly
    predc = []
    pr = []
    sr = []
    for i in range(len(series)):
    #training the model with a train and a validation sets
        epsilon = calcEpsilon(range(len(series[i])),series[i])
        series_pd= pd.DataFrame(series[i])
        series_pd = create_frame(series_pd)
        if is_val == True:
            predicted_series = LP(series_pd.iloc[:200],series_pd.iloc[200:300],epsilon,True)
        elif is_val == False:
            predicted_series = LP(series_pd.iloc[:300],series_pd.iloc[300:],epsilon,True)  
        predc.append(predicted_series)
        pr.append(predicted_series)
        sr.append(series_pd)
        #flag = True
    predc = [sum(x) for x in zip(*predc)]
    mse = mean_squared_error(test_data,predc)
    rmse = sqrt(mse)
    return rmse,predc,pr,sr

def Find_best_k(imfs,lst,clusters,test_data,is_val):
    #finding the chunk_size with minimun RMSE
    RMSE = [] #RMSE score for each chunk_size
    for k in range(len(lst)):
        rmse,predc,pr,sr = ALP_EMD(imfs,lst[k],clusters,test_data,is_val)
        RMSE.append((rmse,lst[k]))
    min_k = min(RMSE, key = lambda t: t[0])
    k = min_k[1]
    #finding the RMSE using the hyperparamter k
    return RMSE,k

for i in range(1,99):
    
   location_data = columns_lst[i].iloc[4000:4600].reset_index(drop=True)
   location_data_overlap = create_frame(location_data)
   imfs, res = find_IMFs_and_res(location_data)    
   validation = location_data_overlap.iloc[200:300].iloc[:,-1].values 
   k = Find_best_k(imfs,[5,15],2,validation,True)
   #print("County number "+str(i))
   #print (k)
   #print (" ")
   test = location_data_overlap.iloc[300:].iloc[:,-1].values 
   rmse,predc,pr,sr = ALP_EMD(imfs,k[1],2,test,False)
   #print ("rmse with chosen K:")
   print(rmse)
   #print(" ") 

############################### EMD+SVR########################################

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for i in range(1,99):
    
    data = columns_lst[i].iloc[4000:4600].reset_index(drop=True)
    
    amplitude = np.array(data.values.flatten().tolist())

    # Define signal
    t = np.arange(0,600)
    s = amplitude
    
    emd = EMD()
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    # _e a plot with all IMFs and residue
    #vis = Visualisation(emd)
    #vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    
    imfs_without_first = [sum(i) for i in zip(*imfs[1:])]
    
    Series = res + imfs_without_first
    
    Location_amplitude = rolling_window(Series,8)
        
    intervals_lst = []  
    for j in range(len(Location_amplitude)):
        extrected_values = Location_amplitude [j].flatten().tolist()
        intervals_lst.append(extrected_values)
            
    Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()
            
    #Splitting the data into train and test sets
    train_location = Location_amplitude.iloc[:300] 
    test_location = Location_amplitude.iloc[300:]
    
    train_x_location = train_location.iloc[:,:-1] #data
    train_y_location = train_location.iloc[:,-1] #labels
    test_x_location = test_location.iloc[:,:-1]
    test_y_location = test_location.iloc[:,-1]
   
    regr = make_pipeline(StandardScaler(), SVR(kernel='rbf',C=1.0, epsilon=0.2))

    regr.fit(train_x_location, train_y_location)
    y_pred_sr = regr.predict(test_x_location)
        
    real = test_y_location.reset_index(drop=True)
            
    mse = mean_squared_error(real,y_pred_sr)
    rmse = sqrt(mse)
    print(rmse)
    
########################## EMD + KRR ##########################################

import warnings
from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rmse_lst = []
for i in range(1,99):
    print(i)
    
    data = columns_lst[i].iloc[4000:4600].reset_index(drop=True)
    
    amplitude = np.array(data.values.flatten().tolist())
    
    epsilon = calcEpsilon(range(len(amplitude)),amplitude)
    
    Location_amplitude = rolling_window(amplitude,8)
    
    intervals_lst = []  
    for j in range(len(Location_amplitude)):
        extrected_values = Location_amplitude [j].flatten().tolist()
        intervals_lst.append(extrected_values)
            
    Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()
    
    real = Location_amplitude.iloc[300:].iloc[:,-1] #real_values

    # Define signal
    t = np.arange(0,600)
    s = amplitude
    
    emd = EMD()
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    # _e a plot with all IMFs and residue
    #vis = Visualisation(emd)
    #vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    
    predc= []
    for imf in range(len(imfs)):
    
        Series = imfs[imf]
        
        Location_amplitude = rolling_window(Series,8)
            
        intervals_lst = []  
        for j in range(len(Location_amplitude)):
            extrected_values = Location_amplitude [j].flatten().tolist()
            intervals_lst.append(extrected_values)
                
        Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()

        predicted_values = LP(Location_amplitude.iloc[:300],Location_amplitude.iloc[300:],epsilon,True)
            
        predc.append(predicted_values)
    
    predc = [sum(x) for x in zip(*predc)]        
                
    mse = mean_squared_error(real,predc)
    rmse = sqrt(mse)
    rmse_lst.append(rmse)

# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)

Average(rmse_lst)

