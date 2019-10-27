# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019
# For this project after testing different parameters, 
# we selected 20 for batch size and 100 for number of epocks and 1e-6 for learning rate
# Also, we run this code for all dataset in GPU




import CrossvalidationDL
import numpy as np


Dataset="Cath3.20.20"
print(Dataset)
Batch_Size=20 
Num_Epochs=100
LR=1e-6
Acc_score,Acc_std,elapsed,lenacc=CrossvalidationDL.DLFold(Dataset,Batch_Size,Num_Epochs,LR)
print("Accuracy_mean: ", np.round(Acc_score,2))
print("Accuracy_std: ", np.round(Acc_std,2))
print("Elapsed time: ",np.round(elapsed,4))
   
	
