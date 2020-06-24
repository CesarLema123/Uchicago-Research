import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors,rdmolfiles,AllChem, DataStructs
from dataManagerResources import PolymerDataManager

"""
    This file test some functionality of the polymerDataManager class for debugging.
"""

#Initialize polymer manager instance
dataManager = PolymerDataManager()
#print(dataManager.df[["polymer_name","smiles"]])

polyIndex = 105
m = Chem.MolFromSmiles(dataManager.df.at[polyIndex,"smiles"])

# Save structure of polymer as an image
Draw.MolToFile(m,"Img/"+dataManager.df.at[polyIndex,"polymer_name"]+"_ind=%i.png"%(polyIndex))

""" Note depictions can be alligned with a common core/substructure to gelp facilitate learning. Note if CNN can learn (with high accuracy and not just "memorizating") to properly label depictions with correct smiles or name identifier then can assume it is capable of learning something about the structure of the 2d molecule representation. That might be enough assume cnn learns the structure of polymer.
"""

# Generate fingerprint from sinlge mol instance
print("---- FP from mol instance: ",FingerprintMols.FingerprintMol(m))

# Generating fingerprints from a collection of smilies (Nx2, col0 = smiles, col1 = identidifer)
smilesdf = dataManager.df[["smiles"]]
smilesdf["identifier"] = range(0,665)
smilesArray = smilesdf.to_numpy()
print("---- FP from smiles data collection: ",FingerprintMols.FingerprintsFromSmiles(smilesArray,1,0))

# Generating fingerprints from a collection of Mol instances (Nx2, col0 = identifier, col1 = mol instance)
print("---- FP from mols: ", FingerprintMols.FingerprintsFromMols( [[1,Chem.MolFromSmiles(dataManager.df.at[0,"smiles"])],[2, Chem.MolFromSmiles(dataManager.df.at[1,"smiles"])]]) )

# Generating fingerprints from individual mol instances.
ms = [Chem.MolFromSmiles('OC(=N)C=C'), Chem.MolFromSmiles('CCCCCCCCNC(=O)C=C'),Chem.MolFromSmiles('COC')]
fps = [Chem.RDKFingerprint(x) for x in ms]


#Covert rdk bit class to bit string
fpBitVector = fps[0]
print("Explicit bit string of fingerprint: \n",fpBitVector.ToBitString())

#Return fingerprints for all polymers in data (assuming smiles is listed)
fp = dataManager.getFingerprints(type = "RDK")
print("RDK style finger print for polymers in raw data: ", fp)
