import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loss_helper import LossManager
"""
This file contains ResNets implementations for Regression
"""
    
class ControlNet0(nn.Module):
    """
    Architecture:
        1x1x10x1x1x10x1
    """
    def __init__(self):
        super(ControlNet0, self).__init__()
        
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 10)
        self.fc3 = nn.Linear(10, 1)
        self.fc4 = nn.Linear(1, 1)
        self.fc5 = nn.Linear(1, 10)
        self.fc6 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

            
class ControlNet1(nn.Module):
    """
    Architecture:
        1x10x10x1x10x10x1
    """
    def __init__(self,numNuerons):
        super(ControlNet1, self).__init__()
        
        self.numNuerons = numNuerons
        
        self.fc1 = nn.Linear(self.numNuerons, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, self.numNuerons)
        self.fc4 = nn.Linear(self.numNuerons, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, self.numNuerons)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class ControlNet2(nn.Module):
    """
    Architecture:
        1x10x10x1x10x10x1
    """
    def __init__(self):
        super(ControlNet2, self).__init__()
        
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.fc4 = nn.Linear(1, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 1)
        self.fc7 = nn.Linear(1, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 1)
        self.fc10 = nn.Linear(1, 10)
        self.fc11 = nn.Linear(10, 10)
        self.fc12 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x =  F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x =  self.fc12(x)
        return x
    
        


class Net0(nn.Module):
    """
    Architecture:
        1xRes[1x10x1]xRes[1x10x1]                 # Res[...(input to next resBlock)]
    Try:
        1xRes[10x10x1]xRes[10x10x1]
    """
    def __init__(self):
        super(Net0, self).__init__()
        
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 10)
        self.fc3 = nn.Linear(10, 1)
        self.fc4 = nn.Linear(1, 1)
        self.fc5 = nn.Linear(1, 10)
        self.fc6 = nn.Linear(10, 1)

    def forward(self, x1):              #xn are inputs to resBlock n
        #resBlock1
        x = F.relu(self.fc1(x1))                # 1->1
        x = F.relu(self.fc2(x))                # 1->10
        x2 = F.relu(self.fc3(x)) + x1           # 10->1
        #resBlock2
        x = F.relu(self.fc4(x2))                # 1->1
        x = F.relu(self.fc5(x))                 # 1->10
        x3 = self.fc6(x) + x2                   # 10->1
        return x3
    
    def loss(self,y,basis):
        invValuesSum = torch.sum(y**(-1))

        return torch.sum(torch.mv(basis,y.reshape(-1))) + invValuesSum
        



class Net1(nn.Module):
    """
    Architecture:
        1xRes[10x10x1]xRes[10x10x1]                 # Res[...(input to next resBlock)]
    """
    def __init__(self):
        super(Net1, self).__init__()
        
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.fc4 = nn.Linear(1, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 1)

    def forward(self, x1):              #xn are inputs to resBlock n
        #resBlock1
        x = F.relu(self.fc1(x1))                # 1->1
        x = F.relu(self.fc2(x))                # 1->10
        x2 = F.relu(self.fc3(x)) + x1           # 10->1
        #resBlock2
        x = F.relu(self.fc4(x2))                # 1->1
        x = F.relu(self.fc5(x))                 # 1->10
        x3 = self.fc6(x) + x2                   # 10->1
        return x3



class Net2(nn.Module):
    """
    Architecture:
        1xRes[10x10x1]xRes[10x10x1]xRes[10x10x1]xRes[10x10x1]                 # Res[...(input to next resBlock)]
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.phaseCoords = None
        
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.fc4 = nn.Linear(1, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 1)
        self.fc7 = nn.Linear(1, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 1)
        self.fc10 = nn.Linear(1, 10)
        self.fc11 = nn.Linear(10, 10)
        self.fc12 = nn.Linear(10, 1)
        
    def forward(self, x1):              #xn are inputs to resBlock n
        #resBlock1
        x = F.relu(self.fc1(x1))                # 1->1
        x = F.relu(self.fc2(x))                 # 1->10
        x2 = F.relu(self.fc3(x)) + x1           # 10->1
        #resBlock2
        x = F.relu(self.fc4(x2))                # 1->1
        x = F.relu(self.fc5(x))                 # 1->10
        x3 = self.fc6(x) + x2                   # 10->1
        #resBlock3
        x = F.relu(self.fc7(x3))                # 1->1
        x = F.relu(self.fc8(x))                 # 1->10
        x4 = F.relu(self.fc9(x)) + x3           # 10->1
        #resBlock4
        x = F.relu(self.fc10(x4))               # 1->1
        x = F.relu(self.fc11(x))                # 1->10
        x5 = self.fc12(x) + x4                  # 10->1
        
        self.phaseCoords = [x1[0].item(),x2[0].item(),x3[0].item(),x4[0].item(),x5[0].item()]
        
        return x3
    """
    def getPhaseCoords()
        return
    """

class Net3(nn.Module):
    """
    Architecture:
        1xRes[10x10x1]xRes[10x10x1]xRes[10x10x1]xRes[10x10x1]xRes[10x10x1]xRes[10x10x1]                 #
    """
    def __init__(self):
        super(Net3, self).__init__()
        
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.fc4 = nn.Linear(1, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 1)
        self.fc7 = nn.Linear(1, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 1)
        self.fc10 = nn.Linear(1, 10)
        self.fc11 = nn.Linear(10, 10)
        self.fc12 = nn.Linear(10, 1)
        self.fc13 = nn.Linear(1, 10)
        self.fc14 = nn.Linear(10, 10)
        self.fc15 = nn.Linear(10, 1)
        self.fc16 = nn.Linear(1, 10)
        self.fc17 = nn.Linear(10, 10)
        self.fc18 = nn.Linear(10, 1)
        
        self.fin1 = nn.Linear(1, 10)
        self.fin2 = nn.Linear(10, 10)
        self.fin3 = nn.Linear(10, 1)
        
    def forward(self, x1):              #xn are inputs to resBlock n
        #resBlock1
        x = F.relu(self.fc1(x1))                # 1->10
        x = F.relu(self.fc2(x))                 # 10->10
        x2 = F.relu(self.fc3(x)) + x1           # 10->1
        #resBlock2
        x = F.relu(self.fc4(x2))                # 1->10
        x = F.relu(self.fc5(x))                 # 10->10
        x3 = self.fc6(x) + x2                   # 10->1
        #resBlock3
        x = F.relu(self.fc7(x3))                # 1->10
        x = F.relu(self.fc8(x))                 # 10->10
        x4 = F.relu(self.fc9(x)) + x3           # 10->1
        #resBlock4
        x = F.relu(self.fc10(x4))               # 1->10
        x = F.relu(self.fc11(x))                # 10->10
        x5 = F.relu(self.fc12(x)) + x4          # 10->1
        #resBlock3
        x = F.relu(self.fc13(x5))               # 1->10
        x = F.relu(self.fc14(x))                # 10->10
        x6 = F.relu(self.fc15(x)) + x5          # 10->1
        #resBlock4
        x = F.relu(self.fc16(x6))               # 1->10
        x = F.relu(self.fc17(x))                # 10->10
        x7 = self.fc18(x) + x6                  # 10->1
        
        return x7
    



def train(net, X, y, epochs, alpha):            # training using MSE loss and batch GD optimizer
    startTime = time.time()                                     # recording train start time
    lossManager = LossManager()                                 # initailizing loss manager
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr = alpha)         # initializing optimizer

    for epoch in range(epochs):                                 # training
        net.zero_grad()
        prediction = net(X)                                     # training output
        loss = criterion(prediction,y)                      # training loss
        loss.backward()                                         # caculates gradients
        optimizer.step()                                        # backprop
        
        lossManager.collect(lossVal = loss.item(), train = True, epoch=True)    # Recording epoch loss
        print("[ Epoch %d/%d ] [ Loss %f ] " % (epoch,epochs, loss.item() ) )
        
    endTime = time.time()
    return lossManager, endTime-startTime

