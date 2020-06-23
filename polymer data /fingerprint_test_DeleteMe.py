import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors,rdmolfiles,AllChem, DataStructs




ms = [Chem.MolFromSmiles('OC(=N)C=C'), Chem.MolFromSmiles('CCCCCCCCNC(=O)C=C'),Chem.MolFromSmiles('COC')]
fps = [Chem.RDKFingerprint(x) for x in ms]

print(type(fps[1]))
print(fps[1].ToBitString())
