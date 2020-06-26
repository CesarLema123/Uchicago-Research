import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors


'''
    This file imports MSAMembrane data and anlyzes it
'''

datafile = "data/MSA Membrane Database - Gas Separation Polymer Membranes.csv"
df = pd.read_csv(datafile)                                          # reading csv to pd.dataframe

print(df.shape)
print(df["Category:"])
