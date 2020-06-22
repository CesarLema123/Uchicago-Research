# Implementation version 2
import numpy as np
import matplotlib.pyplot as plt
    
"""
    Classes for managing loss of a NN during training and testing
"""
    
class LossCollector():
    """ A class for collection of iteration and epoch loss.
    
    """
    def __init__(self):
        self.iterLoss = []                                  # Loss per iteration
        self.epochLoss = []                                 # Loss per Epoch
        self.epochStartInd = None                           # Index of iteration loss at start of Epoch
        self.epochEndInd = None                             # Index of iteration loss at end of Epoch
        
    def collect(self, lossVal, epoch = True):               # Record iter or epoch loss
        if epoch:
            self.epochLoss.append(lossVal)
        else:
            self.iterLoss.append(lossVal)
        
    def collectEpochLoss(self, lossVal):                    # Record Index range of iterloss for an epoch
        self.collect(lossVal, epoch= False)
        if self.epochStartInd is None:
            #self.epochStartInd = np.maximum( 0 , len(self.iterLoss)-1)
            self.epochStartInd = len(self.iterLoss)-1
        self.epochEndInd = len(self.iterLoss)
        return
        
    def avgEpochLoss(self):                                 # Average range of iterloss for an epoch
        self.epochLoss.append( np.mean(self.iterLoss[self.epochStartInd : self.epochEndInd]) )
        self.epochStartInd = None
        self.epochEndInd = None
        return
        
    def getLoss(self, epoch = True):                        # Return last recorded iter or epoch loss
        if epoch:
            return self.epochLoss[-1]
        else:
            return self.iterLoss[-1]
    
class LossManager():
    """ A class for managing collection and visualization of iteration and epoch training and testing loss
    
    """
    def __init__(self):
        self.trainingLoss = LossCollector()
        self.testingLoss = LossCollector()
        
    def collect(self, lossVal, train = True, epoch = True):
        if train:
            self.trainingLoss.collect(lossVal, epoch)
        else:
            self.testingLoss.collect(lossVal, epoch)
        return
        
    def collectEpochLoss(self, lossVal, train = True):
        if train:
            self.trainingLoss.collectEpochLoss(lossVal)
        else:
            self.testingLoss.collectEpochLoss(lossVal)
        return
        
    def avgEpochLoss(self, train = True):
        if train:
            self.trainingLoss.avgEpochLoss()
        else:
            self.testingLoss.avgEpochLoss()
        return
    
    def getLoss(self, train = True, epoch = True):                # Returns last recorded value of loss
        if train:
            return self.trainingLoss.getLoss(epoch)
        else:
            return self.testingLoss.getLoss(epoch)
        
    def plot(self, epochLoss = True, plot = None, save= False,index= 0):
        if plot == "train":
            ''' Plot only training loss '''
        if plot == "test":
            ''' Plot only testing loss '''
            
        if epochLoss:                                           # Loss per epoch
            timeType = "Epoch"
            trainLoss = self.trainingLoss.epochLoss
            testLoss = self.testingLoss.epochLoss
        else:                                                   # Loss per iteration
            timeType = "Iteration"
            trainLoss = self.trainingLoss.iterLoss
            testLoss = self.testingLoss.iterLoss
            
        plt.plot(trainLoss , label='Training Loss')              # Train Loss
        plt.plot(testLoss , label='Testing Loss')                # Test Loss
        plt.title('Loss per '+ timeType)
        plt.legend()
        if save:
            plt.savefig("plt_fullLoss_"+timeType+str(index))
            plt.close()
        else:
            plt.show()
            plt.close()
        
        """ Also print Last 50% of recorded values """
        plt.plot(trainLoss[int(len(trainLoss)/2):] , label='Training Loss')              # Train Loss
        plt.plot(testLoss[int(len(testLoss)/2):] , label='Testing Loss')                # Test Loss
        plt.title('Loss per '+ timeType + ' (Last half of datapoints)')
        plt.legend()
        if save:
            plt.savefig("plt_halfLoss_"+timeType+str(index))
            plt.close()
        else:
            plt.show()
            plt.close()

        


