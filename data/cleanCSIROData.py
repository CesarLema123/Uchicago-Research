import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sys
from rdkit.Chem import Descriptors


'''
    This file imports MSAMembrane data and cleans it
'''

datafile = "rawdata/MSA Membrane Database - Gas Separation Polymer Membranes.csv"
df = pd.read_csv(datafile)                                                      # reading csv to pd.dataframe
gasLabelsInd = 0                                                                # Header row with gas labels
lastHeaderLine = 30                                                             # Last line of Header data
gasLabels = df.loc[gasLabelsInd].to_numpy()                                     # collection of gas labels by column
df = df.loc[lastHeaderLine+1:]                                                  # Remove header data

npData =  df.to_numpy()                                                         # data as np array
NaMatrix = pd.notna( npData )                                                   # boolean matrix for non nan values
columnLabels =  df.columns.to_numpy()                                           # column labels




# --- Save a collection of data matrix with non NaN rows of a column using boolean index of the column ----
cleanData = []                                                                 # collection of np arrays with non na values with respects to column ( identified by index)
for i in range(npData.shape[1]):
    cleanData.append( npData[NaMatrix[:,i] , :])




# --- Save a collection of data matrix with only rows for a single polymer type ---
dataForPolymerType = []                                                      # collection of np arrays matrix data for a specific polyemer type
polymerTypes = np.unique(npData[:,0])                                       # unique polymer types in data

for i in range(len(polymerTypes)):                                          # collect data for each polymer type
    dataForPolymerType.append( npData[ npData[:,0] == polymerTypes[i] , :])



# ----- Save clean data to np file: ------
cleanDataFilename = "CSIROData(clean)"                                        # data filename to save clean data
dataDictionary = {}                                                           # dictionary with keys as gas
for i in range(3 + len(gasLabels[3:18])):
    dataDictionary[gasLabels[i]] = cleanData[i][:,[0,1,2,i]]

print(type(dataDictionary))
np.save(cleanDataFilename, dataDictionary)                                    # saving dictionary to  file


# --- Save data statistics to file: -----
np.set_printoptions(threshold=sys.maxsize, linewidth= sys.maxsize)

filename = "CSIRODataInfo.txt"
outputFile = open(filename,"w")


print("------ Data available --------: \n",file = outputFile)
print("The clean gas permeation data is saved as a dictionary with keys given by the gas name. The values are an numpy array with columns including the polymer type, brief description, polymer, data values. The dictionary is at \n\n                                 %s.npy \n\n"%(cleanDataFilename),file = outputFile)


print("------ Polymer type data --------: \n\nNumber of polymerTypes: %g \n(Note: gases are ordered as He, H2, O2, N2, CO2, CH4, C2H4, C2H6, C3H6, C3H8, C4H8,  n-C4H1, CF4, C2F6, C3F8 )\n"%(len(polymerTypes)) ,file = outputFile)
for i in range(len(dataForPolymerType)):                            # write about polymer type data
    numberGasPValues = np.sum(pd.notna (dataForPolymerType[i][:,3:18]))
    totalValuesPerGas = np.sum( pd.notna (dataForPolymerType[i][:,3:18]),axis = 0)  # in same order ad original data
    gasWithHighestNumValuesIndex = np.argmax( totalValuesPerGas )
    numberOfValues = totalValuesPerGas[gasWithHighestNumValuesIndex  ]
    print( "Polymer Type: %s \n    Number of Polymers: %i \n    Total Number of gas permeation values: %i \n    Total values per gas (ordered as in data): %s \n    Gas with highest number of permeation values: %s \n    Number of values: %i \n\n"%(polymerTypes[i], dataForPolymerType[i].shape[0], numberGasPValues, np.array2string(totalValuesPerGas) , gasLabels[3+gasWithHighestNumValuesIndex], numberOfValues),file = outputFile)
    
print("------- Gas data ---------: \n",file = outputFile)
for i in range(len(cleanData)):                                     # write about values for each gas
    columnData = cleanData[i][:,i]
    try:
        mean = np.mean( columnData.astype(float) )
    except:
        mean = np.nan
    try:
        std = np.std( columnData.astype(float) )
    except:
        std = np.nan

    print( "Column: %s \n    Number of data points: %i \n    Mean: %g \n    std: %g \n"%(gasLabels[i],len(columnData), mean, std ),file = outputFile)
    #print( cleanData[i][:,i])                      # gas permeability values








