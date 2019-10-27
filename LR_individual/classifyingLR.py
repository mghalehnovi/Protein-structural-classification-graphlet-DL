# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019

import numpy as np
import os
import time
import csv
import random
import collections 
from sklearn.linear_model import LogisticRegression
from random import shuffle
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score


def Data_read(data_directory, file):
 raw_data=open(data_directory+'/'+file, 'r')
 data_list=list(raw_data)
 data_ar=np.array(data_list)    # converting to an array
 n=data_ar.size	
 ft_m=data_ar[0].split()             
 ft_ar_m=np.array(ft_m) 
 m=ft_ar_m.size     # the number of columns (features)
 X=np.zeros([n,m-1], dtype="float")                # creating an empty array for Features
 Y=np.empty([n], dtype="S32")
 for i in range(0,n):             #size of data_ar
  ft=data_ar[i].split()     #spliting every row od data_ar
  ft_ar=np.array(ft)           	             # converting to array 
  Y[i]=ft_ar[0]
  for j in range(1,m):                       #creating Features matrix
    X[i,j-1]=ft_ar[j]
 N = np.size(X,0)
 ind_list = [i for i in range(N)]
 random.Random((2)).shuffle(ind_list)
 X=X[ind_list,:] #shuffleing the data
 Y=Y[ind_list,]
 return X, Y

def classifying_LR_l2(data_directory, file):
 X,Y=Data_read(data_directory, file) #read data and shuffling
 num_folds = 10
 model=LogisticRegression()
 Counter_Y=Counter(Y)
 keys=sorted(Counter_Y.keys())
 values=[Counter_Y[key] for key in keys]
 values_of_fold_element=[int(values[i]/num_folds) for i in range(0,len(values))]
 IndexF=[np.where(Y==keys[i])for i in range(0,len(keys))]
 
 Accuracy_list=[]
 for k in range(0,num_folds):
    te_inds_init=[]
    te_inds_init=[np.append(te_inds_init,IndexF[i][0][k*values_of_fold_element[i]:(k+1)*values_of_fold_element[i]]) for i in range(0, len(keys))]
    te_inds  = [int(val) for sublist in te_inds_init for val in sublist]
    tr_inds=[m for m in range(0, len(Y))if m not in te_inds]
    dat_train=X[tr_inds,:]
    labs_train=Y[tr_inds,]
    dat_test=X[te_inds,:]
    labs_test=Y[te_inds]
      
    model.fit(dat_train, labs_train)
    acc_each_fold = model.score(dat_test, labs_test)
    Accuracy_list=np.append(Accuracy_list,acc_each_fold)
 
 return Accuracy_list


listkeys=['matrix-existing-all.txt','matrix-graphlet-3-4.txt','matrix-graphlet-3-5.txt',
        'matrix-normgraphlet-3-4.txt','matrix-normgraphlet-3-5.txt','matrix-normorderedgraphlet-3-4.txt',
        'matrix-normorderedgraphlet-3.txt','matrix-orderedgraphlet-3-4.txt','matrix-orderedgraphlet-3.txt',
        'matrix-sequence.txt','normorderedgraphlet-3-4(K).txt','matrix-CSM.txt','matrix-GIT.txt','matrix-existing-all-pc.txt',
        'matrix-graphlet-3-4-pc.txt','matrix-graphlet-3-5-pc.txt','matrix-normgraphlet-3-4-pc.txt','matrix-normgraphlet-3-5-pc.txt',
        'matrix-normorderedgraphlet-3-4-pc.txt','matrix-normorderedgraphlet-3-pc.txt','matrix-orderedgraphlet-3-4-pc.txt',
        'matrix-orderedgraphlet-3-pc.txt','matrix-sequence-pc.txt','normorderedgraphlet-3-4(K)-pc.txt','matrix-CSM-pc.txt','matrix-GIT-pc.txt']
DictM={}
 
def LRL2(Dataset):
 ROOT_PATH = os.getcwd();
 data_directory = os.path.join(ROOT_PATH, "data/", Dataset)
 
 files = [f for f in os.listdir(data_directory) if f.endswith(".txt")]
 for k in range(0,len(files)):
  start = time.time()
  results=classifying_LR_l2(data_directory, files[k])
  end = time.time()
  elapsed=((end-start)/60) #in minute
  DictM[files[k]]=[results.mean()*100, results.std()*100, elapsed]
  
 name = Dataset+'_LR_PRS.csv'
 namedir='LR_36Dataset_Result'
 Directory_Save=os.path.join(ROOT_PATH, namedir)
 completeName = os.path.join(Directory_Save)
 if not os.path.exists(completeName):
       os.makedirs(completeName)
 with open(Directory_Save+'/'+name,'wb') as csvfile: 
    fieldnames = ['feature','accuracy_mean','accuracy_std','elapsed_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for key in listkeys:
        writer.writerow({'feature':key,'accuracy_mean':"{0:.2f}".format(DictM[key][0]),'accuracy_std':"{0:.2f}".format(DictM[key][1]),'elapsed_time':"{0:.4f}".format(DictM[key][2])})
 return DictM