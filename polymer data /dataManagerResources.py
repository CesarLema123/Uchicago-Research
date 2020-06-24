""" Original copied from https://github.com/peterpaohuang/tapir.git
    edited by cesar lema
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Fingerprints import FingerprintMols


'''
    This file conatins a class for managing polymer data
'''


def calculate_descriptor(smiles, descriptor):
    """ Calculate a single descriptor for single smiles and return descriptor value
    
    Parameters:
        smiles: String
        descriptor: String
    
    Returns:
        descriptor_value: String
            value of generated descriptor based on smiles
    """
    descriptor_method = getattr(Descriptors, descriptor)
    
    try:
        m = Chem.MolFromSmiles(smiles)
        descriptor_value = descriptor_method(m)
        
        return descriptor_value
    
    except Exception as e:
        return np.nan
            
    

class PolymerDataManager:
    """ A class for reading csv data and complementary methods to analyze and add data.
      
    A collection of methods that preprepross data from a csv file.
      
    Attributes:
        df = An pandas dataframe object associated with the class.
        
        experimental_descriptors =
        
        chemical_descriptors =
        
        conversion_formats =
        
    Methods:
        getSmilesIdentifier( polymer_identifier):
            Converts polymer name style identifier to smiles format. Standarizes ( polymer name or smiles ) identifier to only smiles format.
        
        getPolymerDescriptors(polymer_identifier, descriptor_list):
            Returns a dataframe with descriptor values for a single polymer
        
        getCorrelationMatrix(method = "pearson"):
            Returns a correlation matrix of properties in property_list based on given method for correlation
        
        getRDKFingerprints():
            Returns a N (number of polymers in data) row with n (number of bits per fingerprint) column matrix.
                 
        addDescriptors(descriptor_list):
            Adds a column of descriptor values for each descriptor in descriptor_list to the class DataFrame.
        
        (private) propertyExistence(property_list):
            Checks for a property in dataframe or creates new column of property values in the class dataframe for each
            missing property in property_list if possible
        
        plotProperties(property_x = None, property_y = None):
            Plot a scatterplot of two properties against each other
            
        plotMultipleProperties(property_list):
            Plot a pairplot of property_list
        
        propertyCorrelation(property_1, property_2):
            Calculate correlation between two properties based on Pearson correlation
        
        plotPropertyCorrelations(property_list):
            Plot a correlation heatmap of property_list based on Pearson correlation
        
        missingData():
            Plots histogram and prints quantitative data on missing/nan values for the properties in the class dataframe
        
        saveAsCSV(filepath):
            Export current class dataframe to a csv file

    """
    
    def __init__(self, datafile = "data/polymer_db.csv", na_values=["na", ""]):
        """
        Parameters:
            datafile: String, path to csv data file
            na_values: [String], values that are used to signify NaN values
        """
        
        self.df = pd.read_csv(datafile).replace(na_values, np.nan)              # reading csv to pd.dataframe
        if 'Unnamed: 0' in self.df.columns:                                     # removing all no named columns dataframe
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
        self.experimental_descriptors = ["molar_volume", "density", "solubility_parameter","molar_cohesive_energy", "glass_transition_temperature", "molar_heat_capacity", "entanglement_molecular_weight", "refraction_index", "thermal_expansion_coefficient", "repeat_unit_weight", "waals_volume"]
        
        self.chemical_descriptors = ['ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'HeavyAtomMolWt', 'MolWt', 'NumRadicalElectrons', 'NumValenceElectrons', 'BalabanJ', 'BertzCT', 'Ipc', 'HallKierAlpha', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'TPSA']
        
        # self.df.set_index(["polymer_name"], inplace=True)         #set index as polymer name rather than integer - each name is unique

    def __repr__(self):
        properties = "Properties and Descriptors in data are: \n"
        for elem in self.df.columns:
            properties += " " + elem + "\n"
        numPolymer = "Number of Polymers are: \n" + str(len(self.df)) +"\n"
        return properties + numPolymer
        
    def getSmilesIdentifier(self, polymer_identifier):
        """ Converts polymer name style identifier to smiles format. Standarizes ( polymer name or smiles ) identifier to only smiles format.
        
        Parameters:
            polymer_identifier: String, Unique identifier of chemical (either polymer name or smiles)
        
        Return:
            smiles: String, polymer's smiles format
        """
        
        if polymer_identifier in self.df["smiles"].tolist():                    # if polymer_identifier is smiles
            smiles = polymer_identifier
        elif polymer_identifier in self.df["polymer_name"].tolist():            # if polymer_identifier is polymer_name
            smiles = self.df.loc[self.df["polymer_name"] == polymer_identifier]["smiles"].tolist()[0]
        else:
            raise KeyError("Your input did not match any polymer in our database")
        return smiles
        
    """
    def create_input_file(self, polymer_identifier, format, outpath):
        # Not currently using this functionality
        return
    """
    
    def getFingerprints(self,type = "RDK", fpSize = 2048):
        """ Returns a N (number of polymers in data) row with n (number of bits per fingerprint) column matrix. Assumes smiles identifies for polymer is already in rawdata.
        
        """
        if self.propertyExistence(["smiles"]):
            if type == "RDK":
                # --------- complete ----------------
                smiles = self.df["smiles"].to_numpy()[self.df["smiles"].isna() == False]
                returnData = np.zeros((len(smiles),fpSize))
                for ind in range( len(smiles) ):
                    returnData[ind] =  np.array(list(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles[ind])).ToBitString()))
                return returnData
            elif type == "fpModule":
                # --------- Incomplete ----------------
                polyData = self.df[["smiles"]]
                polyData["identifier"] = range(0,len(self.df.index))
                fpData = FingerprintMols.FingerprintsFromSmiles(polyData.to_numpy(),1,0)
                retData = np.zeros((len(fpData),len(list(fpData[0][1].ToBitString()))))

                for i in range(len(retData)):
                    print("----: ",len(list(fpData[i][1].ToBitString())) )
                    #retData[i] = np.array(list(fpData[i][1].ToBitString()))
                return retData
            else:
                raise KeyError("Your input did not match any available finger print types")
        else:
            return None
        
            
            
    def getPolymerDescriptors(self, polymer_identifier, descriptor_list):
        """ returns a dataframe with descriptor values for a single polymer
        
        Parameters:
            polymer_identifier: String, Unique identifier of chemical (either polymer name or smiles)
            descriptor_list: [String], List of descriptors or default returns all descriptors for polymer in class df
        
        Returns:
            single_row_df: DataFrame, dataframe for a single polymer with each column representing a generated descriptor based on descriptor_list
        """
        smiles_indentifier = self.getSmilesIdentifier(polymer_identifier)

        single_row_df = pd.DataFrame()
        for descriptor in descriptor_list:
            single_row_df[descriptor] = [calculate_descriptor(smiles_indentifier, descriptor)]

        #single_row_df.set_index(pd.Index([smiles_indentifier])], inplace=True)              #set index of new dataframe as the unique name of chemical

        return single_row_df
    
    def addDescriptors(self, descriptor_list):
        """ Adds a column of descriptor values for each descriptor in descriptor_list to the class DataFrame.
        
        Parameters:
            descriptor_list: [String], List of descriptors
            
        """
        try:
            for descriptor in descriptor_list:
                if descriptor not in list(self.df):
                    generated_descriptor_series = self.df["smiles"].apply(calculate_descriptor, args=(descriptor,))
                    self.df[descriptor] = generated_descriptor_series        # Adds generated descriptor series to class df
        except:
            raise KeyError("One or multiple of your input properties either do not match any existing\thermo-physical properties in dx.df or property does not match supported chemical descriptors")
        return

#    def add_molecular_structures(self, structure_list):
#        #Not working, missing attributes
#        return

    def propertyExistence(self, property_list):
        """ Checks for a property in dataframe or creates new column of property values in the class dataframe for each missing property in property_list if possible
        
        Parameters:
            property_list: [String], List of properties
            
        Return:
            Returns true if either property was added or already present other wise rasies error.
        """

        try:
            for prop in property_list:
                if prop not in list(self.df):
                    self.addDescriptors([prop])
            return True
        except:
            raise KeyError("One or multiple of your input properties either do not match any existing\
                thermo-physical properties in dx.df or property does not match supported chemical descriptors")
            return False
        
    def plotProperties(self, property_x=None, property_y=None):
        """ Plot a scatterplot of two properties against each other
        
        Parameters:
            property_x: String, property on x-axis
            property_y: String, property on y-axis
        
        """

        self.propertyExistence([property_x, property_y])

        fig, ax = plt.subplots()
        sns.regplot(x=self.df[property_x], y=self.df[property_y])
        fig.tight_layout()
        plt.show()
        return

    def plotMultipleProperties(self, property_list):
        """  Plot a pairplot of property_list
        
        Parameters:
            property_list: [String], list of properties
        
        """
        self.propertyExistence(property_list)

        sns.pairplot(self.df[property_list])
        plt.tight_layout()
        plt.show()
        return

    def propertyCorrelation(self, property_1, property_2):
        """ Calculate correlation between two properties based on Pearson correlation
        
        Parameters:
            property_1: String
            property_2: String
        
        Returns:
            correlation: Float
        """

        self.propertyExistence([property_1, property_2])
        correlation = self.df[property_1].corr(self.df[property_2])
        
        return correlation

    def plotPropertyCorrelations(self,property_list):
        """ Plot a correlation heatmap of properties in property_list based on Pearson correlation
        
        Parameters:
            property_list: [String], list of properties

        """
        self.propertyExistence(property_list)

        fig, ax = plt.subplots()
        corr = self.df[property_list].corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10,
            as_cmap=True), annot=True, ax=ax)
        fig.tight_layout()
        plt.show()
        return

    def getCorrelationMatrix(self, method = "pearson"):
        """ returns a correlation matrix of properties in property_list based on given method for correlation
        
        Parameters:
            method: String, method for computing correlation

        """
        cMatrix = self.df.corr(method= "pearson").to_numpy()
        return cMatrix

    def missingData(self):
        """ Plots histogram and prints quantitative data on missing/nan values for the properties in the class dataframe
        
        """
        fig, ax = plt.subplots()
        nan_df = pd.DataFrame()
        num_nan = [self.df[x].isna().sum() for x in self.df.columns]

        nan_df["num_nan"] = num_nan
        nan_df["column_id"] = self.df.columns
        nan_df.plot(x="column_id", y="num_nan", kind="bar", ax=ax)

        fig.tight_layout()
        plt.show()
        # print number of na values in each column
        print("Number of na values in each column: \n", self.df.isnull().sum())
        return

    def saveAsCSV(self, outpath):
        """ Export current class dataframe to a csv file
        
        Parameters:
            outpath: String, file path to write csv file to

        """
        self.df.to_csv(outpath)
        return

