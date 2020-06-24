import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append("../utils/")

from NNmodels import  linearNet, baseNet0, resBlockNet0, resBlockNet1, train

"""
    This files trains a NN regression model to predict polymer densities
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

XTest = torch.Tensor( X[randInd][int(numSamples*cuttoff):,:])
yTest = torch.Tensor( y[randInd][int(numSamples*cuttoff):,:])

XTrain = torch.Tensor( X[randInd][:int(numSamples*cuttoff),:] )
yTrain = torch.Tensor( y[randInd][:int(numSamples*cuttoff),:] )


#------------------------------------------------------------------
#                        Initializing Model/Training
#------------------------------------------------------------------
numFeatures = XTest.shape[1]
epochs = 50000
alpha = 0.02

#model = baseNet0(numFeatures)
#model = resBlockNet0(numFeatures)
#model = resBlockNet1(numFeatures)
model = linearNet(numFeatures)


lossManager, trainTime = train( model, XTrain, yTrain, epochs, alpha)

#------------------------------------------------------------------
#                    Visualization and Output
#------------------------------------------------------------------

lossManager.plot(save=True)
torch.save(model.state_dict(), "weights_(%i,%i,%i,%g)"%(numSamples, numFeatures, epochs, alpha))

# Poltting train data and predicted values
plt.plot( range(XTrain.shape[0]), yTrain.detach().reshape(-1),"o", label = "data" )
plt.plot( range(XTrain.shape[0]),  model( XTrain ).detach().reshape(-1), "x", label = "NN output" )
plt.title("Train NNOutput vs sample ")
plt.legend()
plt.savefig("Train NNOutput vs samples_(%i,%i,%i,%g).png"%(numSamples, numFeatures, epochs, alpha))
plt.show()
plt.close()


# Poltting test data and predicted values
plt.plot( range(XTest.shape[0]), yTest.detach().reshape(-1),"o", label = "data" )
plt.plot( range(XTest.shape[0]),  model( XTest ).detach().reshape(-1), "x", label = "NN output" )
plt.title("Test NNOutput vs sample ")
plt.legend()
plt.savefig("Test NNOutput vs samples_(%i,%i,%i,%g).png"%(numSamples, numFeatures, epochs, alpha))
plt.show()
plt.close()




# Writing output to file
np.set_printoptions(threshold=sys.maxsize, linewidth= sys.maxsize)
#np.set_printoptions(threshold=sys.maxsize )

filename = "densityNNReg_(%i,%i,%i,%g).txt"%(numSamples, numFeatures, epochs, alpha)
outputFile = open(filename,"w")

print("Training Time: ",trainTime, file = outputFile )
print("Hyperparameters: \n learning rate: %g \n epochs %i"%(alpha,epochs) ,file = outputFile )
print("Train loss value: ",lossManager.getLoss(train = True, epoch = True)  ,file = outputFile )
print("Test loss value: ", torch.nn.MSELoss()( model(XTest).reshape(-1), yTest.reshape(-1) ).item() ,file = outputFile )
outputFile.close()




print("*** Success ***")
