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
from src.clustering import Kmeansclustering

class Preprocessing():
    
    def __init__(self):
        self.trainCsv               = "src/dataset/combined_csv/train/combined.csv"
        self.testCsv                = "src/dataset/combined_csv/test/combined.csv"
        self.predictCsv             = "src/dataset/combined_csv/predict/combined.csv"

        self.preprocessedTrainCsv   = "src/dataset/preprocessed/train"
        self.preprocessedTestCsv    = "src/dataset/preprocessed/test"
        self.preprocessedPredictCsv = "src/dataset/preprocessed/predict"
        self.preprocessedNullCsv    = "src/dataset/preprocessed"
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


    def encodingCategoricalColumnsTraining(self):
        autolog("Mapping started for categorical columns")
        self.dataframe['sex'] = self.dataframe['sex'].map({'F':0 , 'M':1})
        autolog("Mapping for sex column completed")

        for column in self.dataframe.columns:
            if  len(self.dataframe[column].unique())==2:
                self.dataframe[column] = self.dataframe[column].map({'f' : 0, 't' : 1})
        autolog("Mapping for columns with only 2 unique values completed")

        autolog("Mapping completed")

        self.dataframe = get_dummies(self.dataframe, columns=['referral_source'])
        autolog("Applied dummies for referral_source column completed")

        lblEn = LabelEncoder()
        labelencoding =lblEn.fit(self.dataframe['class'])
        self.dataframe['class'] =lblEn.transform(self.dataframe['class'])
        autolog("Label Encoding completed successfully.")

        with open(f"{self.modelsDirs}/encoder.pkl", 'wb') as file:
         pickle.dump(labelencoding, file)
        
    def seperateLabelfeature(self,column_name):
        self.X = self.dataframe.drop(columns = column_name)
        self.Y = self.dataframe[column_name]
        autolog("Label Seperated successfully")
        return self.X, self.Y

    def isnullPresent(self,data,path):
        autolog("checking for null values started")
        self.null_present = False
        #null_present = False
        null_count = data.isna().sum()
        for i in null_count:
            if i>0:
                self.null_present = True
                break
        if (self.null_present):
            dataframe = pandas.DataFrame()
            dataframe['columns'] = data.columns
            dataframe["missing values count"] = numpy.asarray(data.isna().sum())
        dataframe.to_csv(f"{path}/null.csv")
        autolog("Checking for null values completed...")
        return self.null_present


    def imputeNanvalues(self,data):
        autolog("Imputing NaN values started")
        imputer = KNNImputer(n_neighbors=3,weights='uniform', missing_values=nan)
        np_array = imputer.fit_transform(data)
        
        autolog("NaN Values imputation completed successfully")

        autolog("Starting array conversion to dataframe")
        self.new_dataframe = pandas.DataFrame(data = numpy.round(np_array), columns=data.columns)
        autolog("Array to dataframe conversion completed successfully")
        return self.new_dataframe

    # def applyingLogTransformation(self):
    #     autolog("Log Transformation Started")
    #     columns = ['age','tsh','t3','tt4','t4u','fti']
    #     for column in columns:
    #         self.new_dataframe[column] = numpy.log(1 + self.new_dataframe[column])
    #     self.new_dataframe.drop(columns='tsh',inplace=True)
    #     autolog("Dropped TSH column") 
    #     autolog("Log Transformation Completed")
            
    def resampleData(self,X,Y):
        autolog("Resampling of data staryed")
        rdsmple = RandomOverSampler()
        x_sampled,y_sampled  = rdsmple.fit_resample(X,Y)
        self.resampled_dataframe = pandas.DataFrame(data = x_sampled.join(y_sampled), columns = self.dataframe.columns)
        autolog("Resampling of data completed..")
        return x_sampled,y_sampled

    def exportCsv(self,data,path):
        data.to_csv(f"{path}/preprocessed.csv", index=None, header=True)

    
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
    prp.encodingCategoricalColumnsTraining()
    X,Y = prp.seperateLabelfeature('class')
    is_null_present = prp.isnullPresent(X,prp.preprocessedNullCsv)
    print(is_null_present)
    if (is_null_present):
        X = prp.imputeNanvalues(X)
    X,Y = prp.resampleData(X,Y)
    X.to_csv("src/x.csv",index=None,header=True)
    #prp.exportCsv(prp.preprocessedTrainCsv)
    k_means = Kmeansclustering()
    number_of_clusters = k_means.elbowplot(X)
    X = k_means.create_clusters(X,number_of_clusters)
    #print(number_of_clusters)
    prp.exportCsv(X, prp.preprocessedTrainCsv)