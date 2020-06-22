import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
    This files trains a linear regression model to predict a target value
"""

#------------------------------------------------------------------
#                            Loading Data
#------------------------------------------------------------------
#Xtrain = np.linspace(0,2*np.pi,20).reshape(-1,1)
#ytrain = np.sin(Xtrain)                                            # sinusiodal target
ytrain = np.random.multivariate_normal([1,1],[[1,.8],[.8,1]],100)  # multivariate distributed features



print(ytrain[:,0,None].shape)
#------------------------------------------------------------------
#                        Initializing Model/Training
#------------------------------------------------------------------
model = LinearRegression()
#model.fit(Xtrain,ytrain)                                           # fitting sinusiodal
model.fit(ytrain[:,0,None],ytrain[:,1,None])                        # fitting features
    





#------------------------------------------------------------------
#                    Visualization and Output
#------------------------------------------------------------------
#plt.plot(Xtrain,ytrain,label="data")
plt.plot(ytrain[:,0],ytrain[:,1],"o",label="data")
plt.plot(ytrain[:,0].reshape(-1),model.predict(ytrain[:,0,None]).reshape(-1),label="model")
plt.legend()
plt.show()


print("*** Success ***")
