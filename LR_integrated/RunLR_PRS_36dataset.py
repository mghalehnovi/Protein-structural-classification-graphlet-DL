# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019

import classifyingLR
import numpy as np
import os
import csv


ROOT_PATH = os.getcwd()
Datasetlist=["CathP", #Cath first group
    "CathAlpha", "CathBeta", "CathAlphaBeta", #Cath second group
     "Cath1.10", "Cath1.20","Cath2.30", "Cath2.40", "Cath2.60", "Cath2.160", "Cath3.10","Cath3.30", "Cath3.40", # Cath third group
     "Cath2.60.40", "Cath2.60.120", "Cath3.20.20","Cath3.30.390", "Cath3.30.420", "Cath3.40.50", # Cath fourth group
     "SCOPP", # Scop first group
     "SCOPAlpha", "SCOPBeta", "SCOPAlphaBeta", "SCOPAlpha+Beta", "SCOPMD", # Scop second group
     "SCOPa.118", "SCOPb.1", "SCOPc.1", "SCOPc.23", "SCOPc.26", "SCOPc.55", # Scop third group
     "SCOPb.1.1", "SCOPc.1.8", "SCOPc.2.1", "SCOPc.37.1", # Scop fourth group 
	 "Astral"]   #Astral

listkeys=['Concatenate_2Feat.txt','Concatenate_2Feat-pc.txt','Concatenate_12Feat.txt','Concatenate_12Feat-pc.txt']
#listkeys=['Concatenate_2Feat.txt','Concatenate_2Feat-pc.txt','Concatenate_12Feat-pc.txt']

List_all_dict=[]
DictM={}

for i in range(len(Datasetlist)):
    Dataset=Datasetlist[i]
    print(Dataset)
    Dictm=classifyingLR.LRL2(Dataset)
    keys=Dictm.keys()
    AccuracyF=[Dictm[key][0] for key in Dictm.keys()]
    ElapsedF=[Dictm[key][2] for key in Dictm.keys()]
    values=zip(AccuracyF,ElapsedF)
    Dict = dict(zip(keys, values))
    List_all_dict=np.append(List_all_dict,Dict)
    print(set(DictM.keys()) == set(Dict.keys()))
	
    
#Save all acuracy and time results
name_acc= 'Acc_36Data_LR.csv'
name_time='Time_36Data_LR.csv'
namedir='LR_36Dataset_Result'
Directory_Save=os.path.join(ROOT_PATH, namedir)
completeName = os.path.join(Directory_Save)
if not os.path.exists(completeName):
      os.makedirs(completeName)

with open(Directory_Save+'/'+name_acc,'wb') as csvfile:
     fieldnames=np.append(['feature'],Datasetlist)
     writer=csv.writer(csvfile, delimiter=',')
     writer.writerow(fieldnames)
     for key in listkeys:
         row=np.append([key],[np.round(List_all_dict[i][key][0], decimals=2) for i in range(0,len(Datasetlist))])
         writer.writerow(row)

with open(Directory_Save+'/'+name_time,'wb') as csvfile:
     fieldnames=np.append(['feature'],Datasetlist)
     writer=csv.writer(csvfile, delimiter=',')
     writer.writerow(fieldnames)
     for key in listkeys:
         row=np.append([key],[np.round(List_all_dict[i][key][1], decimals=4) for i in range(0,len(Datasetlist))])
         writer.writerow(row)

		 
		 
DictNe={}  #second group cath  
for key in listkeys:
    Acc=[]
    Time=[]
    for i in range(1,4):
        Acc=np.append(Acc,np.round(List_all_dict[i][key][0], decimals=2))
        Time=np.append(Time,np.round(List_all_dict[i][key][1], decimals=4))
    DictNe[key]=[np.mean(Acc), np.std(Acc),np.mean(Time),np.std(Time),sum(Time)] 
name= 'PRS_cath_SG.csv'
namedir='35CathScop_LR'
Directory_Save=os.path.join(ROOT_PATH, namedir)
completeName = os.path.join(Directory_Save)
if not os.path.exists(completeName):
       os.makedirs(completeName)
with open(Directory_Save+'/'+name,'wb') as csvfile:
     fieldnames = ['feature','accuracy_mean','accuracy_std','elapsed_time_mean','elapsed_time_std','elapsed_time_sum']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     writer.writeheader()
     for key in listkeys:
         writer.writerow({'feature':key,'accuracy_mean':"{0:.2f}".format(DictNe.get(key,0)[0]),'accuracy_std':"{0:.2f}".format(DictNe.get(key,0)[1]),'elapsed_time_mean':"{0:.4f}".format(DictNe.get(key,0)[2]),'elapsed_time_std':"{0:.4f}".format(DictNe.get(key,0)[3]),'elapsed_time_sum':"{0:.4f}".format(DictNe.get(key,0)[4])})

DictNe={}  #third group cath  
for key in listkeys:
    Acc=[]
    Time=[]
    for i in range(4,13):
        Acc=np.append(Acc,np.round(List_all_dict[i][key][0], decimals=2))
        Time=np.append(Time,np.round(List_all_dict[i][key][1], decimals=4))
    DictNe[key]=[np.mean(Acc), np.std(Acc),np.mean(Time),np.std(Time),sum(Time)] 
name= 'PRS_cath_TG.csv'
namedir='35CathScop_LR'
Directory_Save=os.path.join(ROOT_PATH, namedir)
completeName = os.path.join(Directory_Save)
if not os.path.exists(completeName):
       os.makedirs(completeName)
with open(Directory_Save+'/'+name,'wb') as csvfile:
     fieldnames = ['feature','accuracy_mean','accuracy_std','elapsed_time_mean','elapsed_time_std','elapsed_time_sum']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     writer.writeheader()
     for key in listkeys:
         writer.writerow({'feature':key,'accuracy_mean':"{0:.2f}".format(DictNe.get(key,0)[0]),'accuracy_std':"{0:.2f}".format(DictNe.get(key,0)[1]),'elapsed_time_mean':"{0:.4f}".format(DictNe.get(key,0)[2]),'elapsed_time_std':"{0:.4f}".format(DictNe.get(key,0)[3]),'elapsed_time_sum':"{0:.4f}".format(DictNe.get(key,0)[4])})
		 
DictNe={}  #fourth group cath  
for key in listkeys:
    Acc=[]
    Time=[]
    for i in range(13,19):
        Acc=np.append(Acc,np.round(List_all_dict[i][key][0], decimals=2))
        Time=np.append(Time,np.round(List_all_dict[i][key][1], decimals=4))
    DictNe[key]=[np.mean(Acc), np.std(Acc),np.mean(Time),np.std(Time),sum(Time)] 
name= 'PRS_cath_FG.csv'
namedir='35CathScop_LR'
Directory_Save=os.path.join(ROOT_PATH, namedir)
completeName = os.path.join(Directory_Save)
if not os.path.exists(completeName):
       os.makedirs(completeName)
with open(Directory_Save+'/'+name,'wb') as csvfile:
     fieldnames = ['feature','accuracy_mean','accuracy_std','elapsed_time_mean','elapsed_time_std','elapsed_time_sum']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     writer.writeheader()
     for key in listkeys:
         writer.writerow({'feature':key,'accuracy_mean':"{0:.2f}".format(DictNe.get(key,0)[0]),'accuracy_std':"{0:.2f}".format(DictNe.get(key,0)[1]),'elapsed_time_mean':"{0:.4f}".format(DictNe.get(key,0)[2]),'elapsed_time_std':"{0:.4f}".format(DictNe.get(key,0)[3]),'elapsed_time_sum':"{0:.4f}".format(DictNe.get(key,0)[4])})

DictNe={}  #second group scop  
for key in listkeys:
    Acc=[]
    Time=[]
    for i in range(20,25):
        Acc=np.append(Acc,np.round(List_all_dict[i][key][0], decimals=2))
        Time=np.append(Time,np.round(List_all_dict[i][key][1], decimals=4))
    DictNe[key]=[np.mean(Acc), np.std(Acc),np.mean(Time),np.std(Time),sum(Time)] 
name= 'PRS_scop_SG.csv'
namedir='35CathScop_LR'
Directory_Save=os.path.join(ROOT_PATH, namedir)
completeName = os.path.join(Directory_Save)
if not os.path.exists(completeName):
       os.makedirs(completeName)
with open(Directory_Save+'/'+name,'wb') as csvfile:
     fieldnames = ['feature','accuracy_mean','accuracy_std','elapsed_time_mean','elapsed_time_std','elapsed_time_sum']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     writer.writeheader()
     for key in listkeys:
         writer.writerow({'feature':key,'accuracy_mean':"{0:.2f}".format(DictNe.get(key,0)[0]),'accuracy_std':"{0:.2f}".format(DictNe.get(key,0)[1]),'elapsed_time_mean':"{0:.4f}".format(DictNe.get(key,0)[2]),'elapsed_time_std':"{0:.4f}".format(DictNe.get(key,0)[3]),'elapsed_time_sum':"{0:.4f}".format(DictNe.get(key,0)[4])})

DictNe={}  #third group scop  
for key in listkeys:
    Acc=[]
    Time=[]
    for i in range(25,31):
        Acc=np.append(Acc,np.round(List_all_dict[i][key][0], decimals=2))
        Time=np.append(Time,np.round(List_all_dict[i][key][1], decimals=4))
    DictNe[key]=[np.mean(Acc), np.std(Acc),np.mean(Time),np.std(Time),sum(Time)] 
name= 'PRS_scop_TG.csv'
namedir='35CathScop_LR'
Directory_Save=os.path.join(ROOT_PATH, namedir)
completeName = os.path.join(Directory_Save)
if not os.path.exists(completeName):
       os.makedirs(completeName)
with open(Directory_Save+'/'+name,'wb') as csvfile:
     fieldnames = ['feature','accuracy_mean','accuracy_std','elapsed_time_mean','elapsed_time_std','elapsed_time_sum']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     writer.writeheader()
     for key in listkeys:
         writer.writerow({'feature':key,'accuracy_mean':"{0:.2f}".format(DictNe.get(key,0)[0]),'accuracy_std':"{0:.2f}".format(DictNe.get(key,0)[1]),'elapsed_time_mean':"{0:.4f}".format(DictNe.get(key,0)[2]),'elapsed_time_std':"{0:.4f}".format(DictNe.get(key,0)[3]),'elapsed_time_sum':"{0:.4f}".format(DictNe.get(key,0)[4])})

DictNe={}  #fourth group scop  
for key in listkeys:
    Acc=[]
    Time=[]
    for i in range(31,35):
        Acc=np.append(Acc,np.round(List_all_dict[i][key][0], decimals=2))
        Time=np.append(Time,np.round(List_all_dict[i][key][1], decimals=4))
    DictNe[key]=[np.mean(Acc), np.std(Acc),np.mean(Time),np.std(Time),sum(Time)] 
name= 'PRS_scop_FG.csv'
namedir='35CathScop_LR'
Directory_Save=os.path.join(ROOT_PATH, namedir)
completeName = os.path.join(Directory_Save)
if not os.path.exists(completeName):
       os.makedirs(completeName)
with open(Directory_Save+'/'+name,'wb') as csvfile:
     fieldnames = ['feature','accuracy_mean','accuracy_std','elapsed_time_mean','elapsed_time_std','elapsed_time_sum']
     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     writer.writeheader()
     for key in listkeys:
         writer.writerow({'feature':key,'accuracy_mean':"{0:.2f}".format(DictNe.get(key,0)[0]),'accuracy_std':"{0:.2f}".format(DictNe.get(key,0)[1]),'elapsed_time_mean':"{0:.4f}".format(DictNe.get(key,0)[2]),'elapsed_time_std':"{0:.4f}".format(DictNe.get(key,0)[3]),'elapsed_time_sum':"{0:.4f}".format(DictNe.get(key,0)[4])})
