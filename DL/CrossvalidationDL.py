# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019

import ClassifyingDL
import os
import numpy as np
import pandas as pd
import timeit
import gc
import sys
import glob
import time
import csv
import random
from random import shuffle
from collections import Counter
from skimage import transform


def Read_data(data_directory, file):
    raw_data=open(data_directory+'/'+file, 'r')
    data_list=list(raw_data)
    data_ar=np.array(data_list)    # converting to an array
    n=data_ar.size
    ft_m=data_ar[1].split()            
    ft_ar_m=np.array(ft_m) 
    m=ft_ar_m.size    # the number of columns (features)
    X=np.zeros([n-1,m-1])                # creating an empty array for Features
    for i in range(1,n):            #size of data_ar
        ft=data_ar[i].split()     #spliting every row od data_ar
        ft_ar=np.array(ft)                        # converting to array 
        for j in range(1,m):                       #creating Features matrix
            X[i-1,j-1]=ft_ar[j]
    return X

def Load_data(Dataset):
    ROOT_PATH=os.getcwd()
    data_directory=os.path.join(ROOT_PATH,"data/",Dataset)
    files=[f for f in os.listdir(data_directory) if f.endswith(".txt")]
    data=[]
    labels=[]
    Size_all=[]
    for i in range(0, len(files)):
        file=files[i]
        label=str(file)[:str(file).find("__")]
        labels=labels+[label]
        X=Read_data(data_directory,file)
        data=data+[X]
        Size_all=Size_all+[X.shape[0]]
    Max_size=max(Size_all)
    Min_size=min(Size_all)
    for i in range(0, len(data)):
          data[i]=transform.resize(data[i], (Min_size,Min_size),mode='reflect',anti_aliasing=True) 
    data=np.asarray(data, dtype=np.float32)
    labels=np.asarray(labels)
    return data, labels
    

def  DLFold(Dataset,Batch_Size,Num_Epochs,LR):
    start_time = timeit.default_timer()
    ROOT_PATH = os.getcwd()
    data_directory=os.path.join(ROOT_PATH,"data/",Dataset)
    Dat, Labs= Load_data(Dataset)
    N = len(Dat)
    ind_list = [i for i in range(N)]
    random.Random(2).shuffle(ind_list)
    Dat  = Dat[ind_list,:,:]
    Labs = Labs[ind_list,]
    Labs=(pd.factorize(Labs)[0]).astype(np.int32)
    number_of_folds=10

    Counter_Y=Counter(Labs)
    keys=sorted(Counter_Y.keys())
    values=[Counter_Y[key] for key in keys]
    values_of_fold_element=[int(values[i]/number_of_folds) for i in range(0,len(values))]
    IndexF=[np.where(Labs==keys[i])for i in range(0,len(keys))]

    num_classes=len(np.unique(Labs))
    Accuracy_list=[]
    for k in range(0,number_of_folds):
        te_inds_init=[]
        te_inds_init=[np.append(te_inds_init,IndexF[i][0][k*values_of_fold_element[i]:(k+1)*values_of_fold_element[i]]) for i in range(0, len(keys))]
        te_inds  = [int(val) for sublist in te_inds_init for val in sublist]
        tr_inds=[m for m in range(0, len(Labs))if m not in te_inds]
        dat_train=Dat[tr_inds,:,:]
        labs_train=Labs[tr_inds,]
        dat_test=Dat[te_inds,:,:]
        labs_test=Labs[te_inds]
                               
        acc_each_fold=ClassifyingDL.DLTF(dat_train,labs_train,dat_test,labs_test,Batch_Size,num_classes,Num_Epochs,LR)
        print("Acc in each fold: ", "{0:.2f}".format(acc_each_fold*100))
        Accuracy_list=np.append(Accuracy_list,acc_each_fold)
    
    elapsed = (timeit.default_timer() - start_time)/60 #in minute
    Acc_mean=np.mean(Accuracy_list)*100
    Acc_std=np.std(Accuracy_list)*100
    lenacc=len(Accuracy_list)
    
    name = Dataset+'_ep'+str(Num_Epochs)+'_bt'+str(Batch_Size)+'_PRS.csv'
    
    namedir='DL_36Dataset'
    Directory_Save=os.path.join(ROOT_PATH, namedir)
    completeName = os.path.join(Directory_Save)
    if not os.path.exists(completeName):
       os.makedirs(completeName)
    with open(Directory_Save+'/'+name,'wb') as csvfile: 
        fieldnames = ['Accuracy_mean','Accuracy_std','Elapsed_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy_mean':"{0:.2f}".format(Acc_mean),'Accuracy_std':"{0:.2f}".format(Acc_std),'Elapsed_time':"{0:.4f}".format(elapsed)})
    

    del dat_train, dat_test
    gc.collect()
    return Acc_mean,Acc_std, elapsed,lenacc