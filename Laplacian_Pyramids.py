############################# IMPORTING ######################################

import numpy as np
import pandas as pd
import sys
import scipy.io
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import normalize
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from numpy.linalg import norm 

##############################################################################

#epsilon_factor - a parameter that controls the width of the Gaussian kernel 
epsilon_factor = 4 

# CalcEpsilon (X,dataList) - Compute  the width of the Gaussian kernel 
#                             based on the given dataset.  

def calcEpsilon(X, dataList):
    dist = []
    for x in X:
        dist.append([])
        for y in X:
            _x = np.array(dataList[x])
            _y = np.array(dataList[y])
            dist[x].append(LA.norm(_x - _y))
    temp = list(dist + np.multiply(np.identity(len(X)) ,max(max(dist))))
    mins = []
    for row in temp:
        small = sys.maxsize
        for el in row:
            if(el < small and el != 0):
                small = el
        mins.append(small)
    return max(mins) * epsilon_factor

# Kernel_matrix - A function that calculates the row-normalized kernel matrix.
# is_zero - setting the diagonal of the kernels at each scale to zero.
def kernel_matrix(dataframe,sigma,is_zero):
    if is_zero == True:
        dataframe = dataframe.iloc[:,:-1].to_numpy() #creating numpy array
        pairwise_dists = pdist(dataframe, 'sqeuclidean') #calcuatlting distances using Pdist
        pairwise_dists = squareform(pairwise_dists) # pdist can be converted to a full distance matrix by using squareform
        kernel = np.exp(-pairwise_dists / sigma**2) 
        np.fill_diagonal(kernel, 0, wrap=True) #fill the main diagonal of the given array with zeros
        normalize_kernel = normalize(kernel, axis=1, norm='l1')
        return (normalize_kernel)
    elif is_zero == False:
        dataframe = dataframe.iloc[:,:-1].to_numpy() #creating numpy array
        pairwise_dists = pdist(dataframe, 'sqeuclidean') #calcuatlting distances using Pdist
        pairwise_dists = squareform(pairwise_dists) # pdist can be converted to a full distance matrix by using squareform
        kernel = np.exp(-pairwise_dists / sigma**2) 
        normalize_kernel = normalize(kernel, axis=1, norm='l1')
        return (normalize_kernel)     

#LP_Train
def LP_Train(dataframe,sigma_0,is_zero):
    f_multiscale = [] #f0,f1,f2...
    f_approx = [] #f0,K1*d1,k2*d2...
    d = [] #d1(f-f0),d2(f-f1),d3(f-f2)...
    lst_sigma = []
    counter = 0
    f = dataframe.iloc[:,-1].to_numpy() #label
    sigma = sigma_0
    while counter <=20:
        if counter == 0:      
            #first iteration, computing kernel matrix (K0), f0 and d1.
            #first iteratin: f0 = approx = f*K0
            k0 = kernel_matrix(dataframe,sigma,is_zero) #kernel matrix
            f0 = k0.dot(f) 
            di = f - f0 #d1
            f_multiscale.append(f0)
            f_approx.append(f0)
            d.append(di)
            lst_sigma.append(sigma_0)
            counter +=1
        else:
            sigma = sigma/2
            lst_sigma.append(sigma)
            ki = kernel_matrix(dataframe,sigma,is_zero)
            approx_i = ki.dot(di)
            f_approx.append(approx_i)
            fi = [sum(i) for i in zip(*f_approx)]
            f_multiscale.append(fi)
            di = f - fi
            d.append(di)
            counter +=1
    err = [norm(i) for i in d]
    return f_multiscale, f_approx,d,err,lst_sigma,f

#LP_Test
def LP(dataframe_train,dataframe_test,sigma_0,is_zero):
    f_multiscale, f_approx,d,err,lst_sigma,f  = LP_Train(dataframe_train,sigma_0,is_zero) #training
    index_min = min(range(len(err)), key=err.__getitem__) #Optimal iteration
    predicted_values = []
    for sample in dataframe_test.values:
        sample = np.array(sample.tolist()).reshape(1,-1)
        pairwise_dists = cdist(sample,dataframe_train,'sqeuclidean').reshape(-1)
        f_new = []
        for i in range(0,index_min+1):
            if i == 0:
                kernel_vector = np.exp(-(pairwise_dists)/(lst_sigma[i]**2))  
                normalize_vector = kernel_vector/kernel_vector.sum(axis=0,keepdims=1)
                if np.isnan([normalize_vector]).sum()>0:
                    normalize_vector = kernel_vector
                else:
                    normalize_vector = normalize_vector
                f0_new = normalize_vector.dot(f)
                f_new.append(f0_new)
            else:
                kernel_vector = np.exp(-(pairwise_dists)/(lst_sigma[i]**2))
                normalize_vector = kernel_vector/kernel_vector.sum(axis=0,keepdims=1)
                if np.isnan([normalize_vector]).sum()>0:
                    normalize_vector = kernel_vector
                else:
                    normalize_vector = normalize_vector
                di_new = normalize_vector.dot(d[i])
                f_new.append(di_new)
        predicted_values.append(sum(f_new))
    return predicted_values