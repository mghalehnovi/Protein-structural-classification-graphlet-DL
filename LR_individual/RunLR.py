# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019

import classifyingLR
import numpy as np


# Datasetlist=["CathP", #Cath first group
    # "CathAlpha", "CathBeta", "CathAlphaBeta", #Cath second group
     # "Cath1.10", "Cath1.20","Cath2.30", "Cath2.40", "Cath2.60", "Cath2.160", "Cath3.10","Cath3.30", "Cath3.40", # Cath third group
     # "Cath2.60.40", "Cath2.60.120", "Cath3.20.20","Cath3.30.390", "Cath3.30.420", "Cath3.40.50", # Cath fourth group
     # "SCOPP", # Scop first group
     # "SCOPAlpha", "SCOPBeta", "SCOPAlphaBeta", "SCOPAlpha+Beta", "SCOPMD", # Scop second group
     # "SCOPa.118", "SCOPb.1", "SCOPc.1", "SCOPc.23", "SCOPc.26", "SCOPc.55", # Scop third group
     # "SCOPb.1.1", "SCOPc.1.8", "SCOPc.2.1", "SCOPc.37.1", # Scop fourth group 
	 # "Astral"]   #Astral 

listkeys=['matrix-existing-all.txt','matrix-graphlet-3-4.txt','matrix-graphlet-3-5.txt',
        'matrix-normgraphlet-3-4.txt','matrix-normgraphlet-3-5.txt','matrix-normorderedgraphlet-3-4.txt',
        'matrix-normorderedgraphlet-3.txt','matrix-orderedgraphlet-3-4.txt','matrix-orderedgraphlet-3.txt',
        'matrix-sequence.txt','normorderedgraphlet-3-4(K).txt','matrix-CSM.txt','matrix-GIT.txt','matrix-existing-all-pc.txt',
        'matrix-graphlet-3-4-pc.txt','matrix-graphlet-3-5-pc.txt','matrix-normgraphlet-3-4-pc.txt','matrix-normgraphlet-3-5-pc.txt',
        'matrix-normorderedgraphlet-3-4-pc.txt','matrix-normorderedgraphlet-3-pc.txt','matrix-orderedgraphlet-3-4-pc.txt',
        'matrix-orderedgraphlet-3-pc.txt','matrix-sequence-pc.txt','normorderedgraphlet-3-4(K)-pc.txt','matrix-CSM-pc.txt','matrix-GIT-pc.txt']

Dataset="Cath3.30.390"
print(Dataset)
print("\n") 
Dictm=classifyingLR.LRL2(Dataset)
for feature in listkeys:
    print("accuracy for "+feature+" is: ", np.round(Dictm[feature][0],2))
    print("elapsed time for "+feature+" is: ", np.round(Dictm[feature][2],4))
    print("\n") 

