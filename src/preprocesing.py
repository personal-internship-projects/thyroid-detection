import numpy
from pandas import read_csv,get_dummies
from numpy import nan
import pandas
import pickle
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from src.logger.auto_logger import autolog
from os.path import isdir
from os import makedirs, read
from src.clustering import Kmeansclustering
from src.model_operations import loadModel, saveModel


class Preprocessing():
    
    def __init__(self):
        self.trainCsv                      = "src/dataset/combined_csv/train/combined.csv"
        self.testCsv                       = "src/dataset/combined_csv/test/combined.csv"
        self.predictCsv                    = "src/dataset/combined_csv/predict/combined.csv"

        self.preprocessedTrainCsv          = "src/dataset/preprocessed/train"
        self.preprocessedTestCsv           = "src/dataset/preprocessed/test"
        self.preprocessedPredictCsv        = "src/dataset/preprocessed/predict"
        self.preprocessedNullCsvTrain      = "src/dataset/preprocessed/train"
        self.preprocessedNullCsvTest       = "src/dataset/preprocessed/test"
        self.preprocessedNullCsvPredict    = "src/dataset/preprocessed/predict"
        self.modelsDirs                    = "src/models"
        self.predModelDirs                 = "src/pred_models" 


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
        self.dataframe.drop(['tsh','tsh_measured','t3_measured','tt4_measured','t4u_measured','fti_measured','tbg_measured','tbg'],axis =1,inplace=True)
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
        print(self.dataframe.columns)
        for column in self.dataframe.columns:
            if  len(self.dataframe[column].unique())<=2:
                self.dataframe[column] = self.dataframe[column].map({'f' : 0, 't' : 1})
            
        autolog("Mapping for columns with only 2 unique values completed")

        autolog("Mapping completed")

        self.dataframe = get_dummies(self.dataframe, columns=['referral_source'])
        autolog("Applied dummies for referral_source column completed")

        lblEn = LabelEncoder()
        labelencoding =lblEn.fit(self.dataframe['class'])
        self.dataframe['class'] =lblEn.transform(self.dataframe['class'])
        autolog("Label Encoding completed successfully.")

        path = f"{self.modelsDirs}/encoder.pkl"
        saveModel(path, lblEn)

    def encodingCategoricalColumnsPrediction(self):
        autolog("Mapping started for categorical columns")
        self.dataframe['sex'] = self.dataframe['sex'].map({'F':0 , 'M':1})
        autolog("Mapping for sex column completed")

        for column in self.dataframe.columns:
            if  len(self.dataframe[column].unique())<=2:
                self.dataframe[column] = self.dataframe[column].map({'f' : 0, 't' : 1})
        autolog("Mapping for columns with only 2 unique values completed")

        autolog("Mapping completed")

        self.dataframe = get_dummies(self.dataframe, columns=['referral_source'])
        autolog("Applied dummies for referral_source column completed")
    

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
        model  = imputer.fit(data)
        np_array = model.transform(data)
        
        autolog("NaN Values imputation completed successfully")

        autolog("Saving Imputer model...")
        saveModel("src/models/Imputer.pkl", model)
        autolog("Imputer model saved successfully.")

        return pandas.DataFrame(data = numpy.round(np_array, 4), columns=data.columns)


    def imputeNaNValuesOnTestAndPredict(self, data):
        model = loadModel("src/models/Imputer.pkl")
        imputedData = model.transform(data)
        return pandas.DataFrame(data = numpy.round(imputedData, 4), columns=data.columns)


    def removeOutlier(self, df):
        #outColumns = {"age": .9995, "tt4":""}

        q = df['age'].quantile(.01)
        df_new = df[df['age'] > q]

        q = df_new['age'].quantile(.9995)
        df_new = df_new[df_new['age'] < q]
        print(f"age: {df_new}")

        q = df_new['tt4'].quantile(.98)
        df_new = df_new[df_new['tt4'] < q]

        q = df_new['fti'].quantile(.96)
        df_new = df_new[df_new['fti'] < q]

        q = df_new['t3'].quantile(.99)
        df_new = df_new[df_new['t3'] < q]

        q = df_new['t4u'].quantile(.92)
        df_new = df_new[df_new['t4u'] < q]

        q = df_new['t4u'].quantile(.00245)
        df_new = df_new[df_new['t4u'] > q]


        i = df_new[df_new['tt4']==-5.199337582605575].index
        df_new.drop(i,inplace=True)

        i = df_new[df_new['t3']==-5.199337582605575].index
        df_new.drop(i,inplace=True)

        i = df_new[df_new['age']==3.09400722556956].index
        df_new.drop(i,inplace=True)


        self.dataframe = df_new.copy()


    def seperateLabelfeature(self,column_name):
        self.X = self.dataframe.drop(columns = column_name)
        self.Y = self.dataframe[column_name]
        autolog("Label Seperated successfully")
        return self.X, self.Y


    def resampleData(self,path,X,Y):
        autolog("Resampling of data started")
        rdsmple = RandomOverSampler(random_state=42)
        x_sampled,y_sampled  = rdsmple.fit_resample(X,Y)
        self.resampled_dataframe = pandas.DataFrame(data = x_sampled.join(y_sampled), columns = self.dataframe.columns, index=None)
        self.resampled_dataframe.to_csv(f"{path}/preprocessed.csv")    
        
        ## for checking if data is properly resampled or not
        
        x_sampled.to_csv(f"{path}/preprocessed_X.csv", index=None, header=True)
        y_sampled.to_csv(f"{path}/preprocessed_Y.csv", index=None, header=True)

        autolog("Resampling of data completed..")
        return  read_csv(f"{path}/preprocessed_X.csv"), read_csv(f"{path}/preprocessed_Y.csv")


    def LogTransformer(self, data):
        """

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        columns = ['age','t3','tt4','t4u','fti']
        for cols in columns:
            data[cols] = numpy.log(1 + data[cols])
        print(data)
        return numpy.round(data, 4)


    def exportCsv(self,data,path):
        data.to_csv(f"{path}/preprocessed.csv", index=None, header=True)

    
if __name__ == '__main__':
    # importing database operations class
    """from src.database_operations import CassandraOperations
    #created obejct for database class
    ops = CassandraOperations()
    #Intialized Connection to database
    ops.databaseConnection()
    #Fetching data from database
    ops.fetch(ops.combinedTrain, "train",  ops.schemaPath)
   """ #Importing preprocessing class
    prp = Preprocessing()
    #Reading csv from the path
    prp.readCsv(prp.trainCsv)
    #Creating directory for data after preprocessing
    prp.createPreprocessedDirectory()
    #Dropping unnecessary columns
    prp.dropUnnecessaryColumns()
    #Replacing '?' values with NaN 
    prp.replaceWithNan()
    # Handling categorical features and  converting categorical features to numerical
    prp.encodingCategoricalColumnsTraining()
    #Scaling the data to handle skewness of the dataset and dropping useless column "TSH"
    #Checking for null in dataset
    is_null_present = prp.isnullPresent(prp.dataframe,prp.preprocessedNullCsv)
    print(is_null_present)
    #If null present then replace NaN with values predicted by KNN algorithm 
    if (is_null_present):
        prp.imputeNanvalues(prp.dataframe)
    
    prp.removeOutlier()
    # seperating label and features columns
    X,Y = prp.seperateLabelfeature('class')
    print(Y.unique())
    X = prp.quantileTransformer(X)
    # HAndling Imbalanced dataset
    X,Y = prp.resampleData(prp.preprocessedTrainCsv,X,Y)
    
    #Creating obj for Kmeans clustering class
    k_means = Kmeansclustering()
    #Getting the optimal values of k for kmeans clustering using elbowplot
    number_of_clusters = k_means.elbowplot(X)
    #Creating clusters for each datapoint
    #X.to_csv("/home/gamer/Downloads/fff.csv")
  #  X = read_csv("/home/gamer/Downloads/ff.csv")
    k_means.silhoutee_scores(X)
    k_means.scores_clustering()
    X_clusters = k_means.create_clusters(X,number_of_clusters)
    #autolog(f"number of clusters are: {number_of_clusters}")
    #Exporting the dataset for self reference
    """df12 = X_clusters.join(Y)
    autolog("start")
    prp.exportCsv(df12, prp.preprocessedTrainCsv)
    autolog("finished")
    """#from sklearn.metrics import silhouette_samples, silhouette_score
    # silhouette_avg = silhouette_score(X, X_clusters['Cluster'])
    # print("************************************")
    # print(number_of_clusters)
    # print(silhouette_avg)
    # print("************************************")