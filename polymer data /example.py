from  polymerDataManager import PolymerDataManager
import matplotlib.pyplot as plt
import numpy as np
from rdkit import DataStructs

""" Testing Functionality of PolymerDataManager class
"""

#------------------------------------------------------------------
#                       Initializing class
#------------------------------------------------------------------
datafile = "data/polymer_db.csv"
dataManager = PolymerDataManager(datafile)

#------------------------------------------------------------------
#                      Testing Functionality
#------------------------------------------------------------------
# print properties and descriptors in data
print(dataManager)

# Get descriptors dataframe of specified polymer
descriptors = dataManager.getPolymerDescriptors("Poly(acrylamide)",['HallKierAlpha','ExactMolWt'])
print("Descriptors dataframe for Poly(acrylamide): \n", descriptors, "\n")
    
# Add values for specified descriptors to instance dataframe
dataManager.addDescriptors(['HallKierAlpha','ExactMolWt','BalabanJ', 'BertzCT', 'Ipc', 'HallKierAlpha', 'MolLogP', 'MolMR'])

# Plot 2 properties of polymers in data
dataManager.plotProperties("molar_volume","ExactMolWt")

# Plot multiple properties against each other.
dataManager.plotMultipleProperties( ["molar_volume","ExactMolWt","HallKierAlpha"] )

# Compute pearson correlation for 2 properties
correlation = dataManager.propertyCorrelation("molar_volume","ExactMolWt")
print("Pearson correlation between molar_volume and ExactMolWt: ", correlation,"\n")

# plot correlation for multiple given properties
dataManager.plotPropertyCorrelations(["molar_volume","ExactMolWt","HallKierAlpha"])
#print(dataManager.df.corr(method= "pearson"))                                             # alternate pandas implementation

# get correlation matrix for multiple given properties
matrix = dataManager.getCorrelationMatrix()
print("Data correlation matrix: \n",matrix, "\n")

# quantitative description of missing values
dataManager.MissingData()

# save current data in dataManager
dataManager.saveAsCSV("data/example.csv")





