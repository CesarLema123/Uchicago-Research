from  dataManagerResources import PolymerDataManager
import matplotlib.pyplot as plt
import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import Descriptors

#from pylatexenc.latexencode import unicode_to_latex         # for saving text in a latex format

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

""" miscellaneous Code for testing rdkit descriptors available 
#print(unicode_to_latex(Descriptors.descList[0][0])              # for saving text in a latex format
#print(help(Descriptors))
#print(Descriptors.descList)
#print(dataManager.getPolymerDescriptors("OC(=N)C=C",["fr_pyridine"]))
#print(dir(Descriptors))

i =0
for elem in dir(Chem.rdMolDescriptors):
    print(i, elem)
    i+=1

descriptor_method = getattr(Descriptors, "fr_pyridine")
print("----: ",descriptor_method,type(descriptor_method))
print(Descriptors.fr_prisulfonamd(Chem.MolFromSmiles("OC(=N)C=C")) )
print(Descriptors.Autocorr2D(Chem.MolFromSmiles("OC(=N)C=C")))

quit()
"""

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
dataManager.missingData()

# save current data in dataManager
dataManager.saveAsCSV("data/example.csv")





