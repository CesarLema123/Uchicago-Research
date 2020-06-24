import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
import sys
sys.path.append("../polymer data/")

from dataManagerResources import PolymerDataManager

"""
    This files trains a linear regression model using sklearn to predict polymer densities
"""

#------------------------------------------------------------------
#                            Loading Data
#------------------------------------------------------------------

X = np.load("densityDataset.npy",allow_pickle=True).item()["XTrain"]
y = np.load("densityDataset.npy",allow_pickle=True).item()["yTrain"]

"""
X2 = np.zeros((X.shape[0],X.shape[1]))
for elem in X:
    lst = []
    curr = ""
    for i in range(len(elem)):
        if i%2 == 0 and i!=0:
            lst.append(int(curr,2))
            curr = ""
        else:
            curr += str(int(elem[i]))
    print(lst)
X = X2
"""

cuttoff = .7
numSamples = len(X)
randInd = np.random.permutation(numSamples)

XTest = X[randInd][int(numSamples*cuttoff):,:]
yTest = y[randInd][int(numSamples*cuttoff):,:]

XTrain = X[randInd][:int(numSamples*cuttoff),:]
yTrain = y[randInd][:int(numSamples*cuttoff),:]


#------------------------------------------------------------------
#                        Initializing Model/Training
#------------------------------------------------------------------
model = LinearRegression()
model.fit(XTrain,yTrain)                                           # fitting sinusiodal
    



#------------------------------------------------------------------
#                    Visualization and Output
#------------------------------------------------------------------

# Training plot
plt.plot(range(len(yTrain)),yTrain,"o",label="data")
plt.plot(range(len(yTrain)),model.predict(XTrain),"x",label="model")
plt.title("Training results")
plt.legend()
plt.savefig("Train results.png")
plt.show()
plt.close()

# Testing plot
plt.plot(range(len(yTest)),yTest,"o",label="data")
plt.plot(range(len(yTest)),model.predict(XTest).reshape(-1),"x",label="model")
plt.title("Testing results")
plt.legend()
plt.savefig("Testing results.png")
plt.show()
plt.close()

#Writing output to file
np.set_printoptions(threshold=sys.maxsize, linewidth= sys.maxsize)

filename = "densityLinearReg.txt"
outputFile = open(filename,"w")
print("Model score: ",model.score(XTest,yTest),file = outputFile)
outputFile.close()



print("*** Success ***")
