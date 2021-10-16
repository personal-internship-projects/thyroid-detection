import numpy
from pandas import read_csv,get_dummies
from numpy import nan
import pandas
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
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
        self.modelsDirs             = "src/models"


    def createPreprocessedDirectory(self):
        if not isdir(self.preprocessedPredictCsv):
            makedirs(self.preprocessedPredictCsv)

        if not isdir(self.preprocessedTestCsv):
            makedirs(self.preprocessedTestCsv)

        if not isdir(self.preprocessedTrainCsv):
            makedirs(self.preprocessedTrainCsv)

        if not isdir(self.modelsDirs):
            makedirs(self.modelsDirs)


    def readCsv(self,path):
        self.dataframe = read_csv(f'{path}')
        autolog("Read CSV Successfully")


    def dropUnnecessaryColumns(self):
        self.dataframe.drop(columns="id",inplace=True)
        self.dataframe.drop(['tsh_measured','t3_measured','tt4_measured','t4u_measured','fti_measured','tbg_measured','tbg'],axis =1,inplace=True)
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
        labelencoding =lblEn.fit(self.dataframe['class'])
        self.dataframe['class'] =lblEn.transform(self.dataframe['class'])
        autolog("Label Encoding completed successfully.")

        with open(f"{self.modelsDirs}/encoder.pkl", 'wb') as file:
         pickle.dump(labelencoding, file)

    def imputeNanvalues(self):
        autolog("Imputing NaN values started")
        imputer = KNNImputer(n_neighbors=3,weights='uniform', missing_values=nan)
        np_array = imputer.fit_transform(self.dataframe)
        
        autolog("NaN Values imputation completed successfully")

        autolog("Starting array conversion to dataframe")
        self.new_dataframe = pandas.DataFrame(data = numpy.round(np_array), columns=self.dataframe.columns)
        autolog("Array to dataframe conversion completed successfully")

    def applyingLogTransformation(self):
        autolog("Log Transformation Started")
        columns = ['age','tsh','t3','tt4','t4u','fti']
        for column in columns:
            self.new_dataframe[column] = numpy.log(1 + self.new_dataframe[column])
        self.new_dataframe.drop(columns='tsh',inplace=True)
        autolog("Dropped TSH column") 
        autolog("Log Transformation Completed")
            
    def resampleData(self):
        x = self.new_dataframe.drop(['class'],axis=1)
        y = self.new_dataframe['class']
        rdsmple = RandomOverSampler()
        x_sampled,y_sampled  = rdsmple.fit_resample(x,y)
        self.resampled_dataframe = pandas.DataFrame(data = x_sampled, columns = x.columns)

    def exportCsv(self, path):
        self.resampled_dataframe.to_csv(f"{path}/preprocessed.csv", index=None, header=True)

    
if __name__ == '__main__':
    from src.database_operations import CassandraOperations
    ops = CassandraOperations()
    ops.databaseConnection()
    ops.fetch(ops.combinedTrain, "train",  ops.schemaPath)

    prp = Preprocessing()
    prp.readCsv(prp.trainCsv)
    prp.createPreprocessedDirectory()
    prp.dropUnnecessaryColumns()
    prp.replaceWithNan()
    prp.mappingCategoricalColumns()
    prp.getDummies()
    prp.labelEncoding()
    prp.imputeNanvalues()
    prp.applyingLogTransformation()
    prp.resampleData()
    prp.exportCsv(prp.preprocessedTrainCsv)