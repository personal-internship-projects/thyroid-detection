from pandas import read_csv,get_dummies
from numpy import nan
from sklearn.preprocessing import LabelEncoder
from src.logger.auto_logger import autolog
from os.path import isdir
from os import makedirs

class Preprocessing():
    
    def __init__(self):
        self.trainCsv               = "src/dataset/combined_csv/train/combined.csv"
        self.testCsv                = "src/dataset/combined_csv/test/combined.csv"
        self.predictCsv             = "src/dataset/combined_csv/predict/combined.csv"

        self.preprocessedTrainCsv   = "src/dataset/preprocessed/train"
        self.preprocessedTestCsv    = "src/dataset/preprocessed/test"
        self.preprocessedPredictCsv = "src/dataset/preprocessed/predict"


    def createPreprocessedDirectory(self):
        if not isdir(self.preprocessedPredictCsv):
            makedirs(self.preprocessedPredictCsv)

        if not isdir(self.preprocessedTestCsv):
            makedirs(self.preprocessedTestCsv)

        if not isdir(self.preprocessedTrainCsv):
            makedirs(self.preprocessedTrainCsv)


    def readCsv(self,path):
        self.dataframe = read_csv(f'{path}')
        autolog("Read CSV Successfully")


    def dropUnnecessaryColumns(self):
        self.dataframe.drop(columns="id",inplace=True)
        self.dataframe.drop(['tsh_measured','t3_measured','tt4_measured','t4u_measured','fti_measured','tbg_measured'],axis =1,inplace=True)
        autolog("Dropped UnnecessaryColumns")
    

    def replaceWithNan(self):
        autolog("Replace starting")
        for column in self.dataframe.columns:
            count = self.dataframe[column][self.dataframe[column]=='?'].count()
            if count!=0:
                self.dataframe[column] = self.dataframe[column].replace('?',nan)    
        autolog("Replaced '?' with nan")


    def mappingCategoricalColumns(self):
        autolog("Mapping started for categorical columns")
        self.dataframe['sex'] = self.dataframe['sex'].map({'F':0 , 'M':1})
        autolog("Mapping for sex column completed")

        for column in self.dataframe.columns:
            if  len(self.dataframe[column].unique())==2:
                self.dataframe[column] = self.dataframe[column].map({'f' : 0, 't' : 1})
        autolog("Mapping for columns with only 2 unique values completed")

        autolog("Mapping completed")

        
    def getDummies(self):
        self.dataframe = get_dummies(self.dataframe, columns=['referral_source'])
        autolog("Applied dummies for referral_source column completed")


    def labelEncoding(self):
        lblEn = LabelEncoder()
        self.dataframe['class'] =lblEn.fit_transform(self.dataframe['class'])
        autolog("Label Encoding completed successfully.")


    def exportCsv(self, path):
        self.dataframe.to_csv(f"{path}/preprocessed.csv", index=None, header=True)
    
    
