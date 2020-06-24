import numpy as np
import sys
sys.path.append("../polymer data/")

from dataManagerResources import PolymerDataManager


"""
    This file preprocess raw data and save as either numpy or torch data file.
"""

#--------------------------------------------------------------------------------
#                Loading, cleaning,and saving a clean density dataset
#--------------------------------------------------------------------------------

rawDataMan = PolymerDataManager(datafile = "../polymer data/data/polymer_db.csv")       # Init raw data manager

#print(rawDataMan.missingData())                                                        # Show stats of nan values in raw data
XTrain = rawDataMan.getFingerprints()                                                   # Get fp for polymers in raw data with smiles identifier
yTrain = rawDataMan.df["density"][rawDataMan.df["smiles"].isna() == False].to_numpy().reshape(-1,1)         # Get only raw density input for polymers with smiles identifies as a np array

XTrain = XTrain[np.isnan(yTrain).reshape(-1) == False]                                  # Remove samples with no/np.nan target values
yTrain = yTrain[np.isnan(yTrain).reshape(-1) == False]                                  # Remove samples with no/np.nan target values

np.save("densityDataset",{ "XTrain": XTrain, "yTrain": yTrain})
