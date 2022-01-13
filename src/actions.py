import pandas as pd
from pandas.core.frame import DataFrame
import src.Data_Validation as dv
import src.database_operations as dboc
import src.preprocesing as prp
from src.model_operations import loadModel
from src.clustering import Kmeansclustering
from src.logger.auto_logger import autolog
from os import read
from pandas.io.parsers import read_csv
from sklearn import cluster
import src.File_Type_Validation as fv
import src.Data_Validation as dv
from sklearn.model_selection import train_test_split
import src.database_operations as dboc
from src.model_operations import saveModel
from src import model_finder as mf
import warnings
warnings.filterwarnings("ignore")

class MLoperations:
    
    def train_model(self):
        
        ''' x = fv.File_Type_Validation("./src/dataset")
        x.createCsvDir()
        x.convertToCsv()
        zz =dv.DataValidation()
        zz.makeFinalCsvDirectory(zz.finalCsvTrain)
        zz.makeFinalCsvDirectory(zz.finalCsvTest)
        z,g,dic = zz.verifyingSchema(zz.schemaPath)
        a= zz.validateColumnLength(g,zz.goodCsvPath)
        b = zz.validateMissingValuesInWholeColumn(zz.goodCsvPath)
        d= zz.getColumnName(zz.schemaPath)
        e=zz.addColumnNames(d, zz.goodCsvPath)
        f = zz.addQuotesToString(dic)
        db = dboc.CassandraOperations()
        db.databaseConnection()
        db.createPreprocessedCsvDirectory(db.combinedTrain)
        db.deleteTable('train')
        db.createTable('train', db.schemaPath)
        db.insertValidatedData(db.finalCsvTrain, "train", db.schemaPath)
        db.fetch(db.combinedTrain, "train",  db.schemaPath)'''


        pre = prp.Preprocessing()

        pre.createPreprocessedDirectory()
        pre.readCsv(pre.trainCsv)
        pre.dropUnnecessaryColumns()
        pre.replaceWithNan()
        pre.encodingCategoricalColumnsTraining()
        isNullPresent = pre.isnullPresent(pre.dataframe,pre.preprocessedNullCsvTrain)

        X_train, Y_train = pre.seperateLabelfeature('class')
        if (isNullPresent):
            X_train = pre.imputeNanvalues(X_train)

        X_train, Y_train = pre.resampleData(pre.preprocessedTrainCsv, X_train, Y_train)

        # After resampling data, we are separating X, Y 
        # for applying logarithmic transformer

        autolog("Applying Logarithmic Transformer...")
        X_train = pre.LogTransformer(X_train)
        autolog("Log Transformer applied")

        tmp_dataframe = X_train.join(Y_train)

        # After applying data transformation on numeric column 

        print(tmp_dataframe['class'].unique())

        # After doing all the steps above, we are separating X and Y again
        # This step is done for clustering

        X_train = tmp_dataframe.drop(columns = ["class"])
        Y_train = tmp_dataframe["class"]

        autolog("Clustering started.")
        K_Mean = Kmeansclustering() 

        numberOfClusters = K_Mean.elbowplot(X_train)

        K_Mean.silhoutee_scores(X_train)
        K_Mean.scores_clustering()
        
        X_train_clusters = K_Mean.create_clusters(X_train, numberOfClusters)
        autolog(f"number of clusters are: {numberOfClusters}")

        dfTransport = X_train_clusters.join(Y_train)
        autolog("Starting export ...")
        pre.exportCsv(dfTransport, pre.preprocessedTrainCsv)
        autolog("Done.")

        print(numberOfClusters)

        ## Training started

        finalDataframeTrain = read_csv(f"{pre.preprocessedTrainCsv}/preprocessed.csv")
        finalDataframeTest = read_csv(f"{pre.preprocessedTestCsv}/preprocessed.csv")

        clusterID = finalDataframeTrain['Cluster'].unique()

        for id in clusterID:
            autolog(f"Training started for Cluster {id}.")
            ## Separating data based on cluster
            clusterDataTrain     = finalDataframeTrain[finalDataframeTrain['Cluster'] == id]
            
            ## Prepare the feature and Label columns
            x = clusterDataTrain.drop(['class', 'Cluster'], axis=1)
            y = clusterDataTrain['class']

            ## Splitting the data into training and testing data
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
            
            model = mf.ModelFinder(x_train,y_train,x_test,y_test)
            model.getBestparamsDecisionTree()
            model.getBestparamsRandomForest()
            model.getBestparamsKNN()
            modelName, predModel = model.getBestModel()
            saveModel(f"{pre.modelsDirs}/{modelName}_{id}.pkl", predModel)
            saveModel(f"{pre.predModelDirs}/{id}.pkl", predModel)
            autolog(f"Training for cluster {id} completed successfully")
            
    
    def test_model(self):
    
        """db = dboc.CassandraOperations()
        db.databaseConnection()
        db.createPreprocessedCsvDirectory(db.combinedTest)
        db.deleteTable("test")
        db.createTable('test',db.schemaPath)
        db.insertValidatedData(db.finalCsvTest, "test", db.schemaPath)
        db.fetch(db.combinedTest,'test', db.schemaPath)"""

        pre = prp.Preprocessing()
        pre.createPreprocessedDirectory()
        pre.readCsv(pre.testCsv)
        pre.dropUnnecessaryColumns()
        pre.replaceWithNan()
        pre.encodingCategoricalColumnsTraining()
        is_null_present = pre.isnullPresent(pre.dataframe, pre.preprocessedNullCsvTest)

        X_test, Y_test = pre.seperateLabelfeature('class')
        predictedData = X_test

        if (is_null_present):
            X_test = pre.imputeNaNValuesOnTestAndPredict(X_test)

        ## After resampling data, we are separating X, Y 
        ## for applying quantile transformer

        autolog("Applying Logarithmic Transformer...")
        finalDataframeTest = pre.LogTransformer(X_test)
        autolog("Log Transformer applied")

        try:
            K_Mean = loadModel("src/models/kmeans-clustering.pkl") 

            finalDataframeTest["Cluster"] = K_Mean.predict(finalDataframeTest)
            predictedData["Cluster"] = finalDataframeTest["Cluster"]
            
        except Exception as e:
            autolog(f"An error occured {e}")


        # testing started
        autolog("Prediction started")
        clusterID = finalDataframeTest['Cluster'].unique()

        for id in clusterID:
            autolog(f"Training started for Cluster {id}.")
            ## Separating data based on cluster
            clusterDataTest = finalDataframeTest[finalDataframeTest['Cluster'] == id]
            
            # ## Prepare the feature and Label columns
            x = clusterDataTest.drop(['Cluster'], axis=1)
        
            # ## Load the model      
            pred_model = loadModel(f"{pre.predModelDirs}/{id}.pkl")
            predictedData.loc[predictedData['Cluster'] == id, 'class'] = pred_model.predict(x)
            

        decoder = loadModel("src/models/encoder.pkl")

        predictedData["class"] = decoder.inverse_transform(predictedData["class"].astype(int))
        predictedData.drop(['Cluster'], axis=1, inplace=True)

        autolog("Saving predicted data...")
        pre.exportCsv(predictedData, pre.preprocessedTestCsv)
        autolog("Data saved")
        
    def prediction(self):
        zz = dv.DataValidation()
        zz.makeFinalCsvDirectory(zz.finalCsvPredict)
        z, g, dic = zz.verifyingSchema(zz.schemaPathPredict)
        a = zz.validateColumnLength(g, zz.predictCsvPath)
        b = zz.validateMissingValuesInWholeColumn(zz.predictCsvPath)
        d = zz.getColumnName(zz.schemaPathPredict)
        e = zz.addColumnNames(d, zz.predictCsvPath)
        f = zz.addQuotesToStringPredict(dic)


        """db = dboc.CassandraOperations()
        db.databaseConnection()
        db.createPreprocessedCsvDirectory(db.combinedPredict)
        db.deleteTable('predict')
        db.createTable('predict', db.schemaPathPredict)
        db.insertValidatedData(db.finalCsvPredict, "predict", db.schemaPathPredict)
        db.fetch(db.combinedPredict, "predict", db.schemaPathPredict)"""

        pre = prp.Preprocessing()
        pre.createPreprocessedDirectory()
        pre.readCsv(pre.predictCsv)
        pre.dropUnnecessaryColumns()
        pre.replaceWithNan()
        pre.encodingCategoricalColumnsPrediction()
        is_null_present = pre.isnullPresent(
            pre.dataframe, pre.preprocessedNullCsvPredict)

        predictedData, finalDataframePredict = pre.dataframe, pre.dataframe
        if (is_null_present):
            finalDataframePredict = pre.imputeNaNValuesOnTestAndPredict(finalDataframePredict)

        autolog("Applying Logarithmic Transformer...")
        finalDataframePredict = pre.LogTransformer(finalDataframePredict)
        autolog("Log Transformer applied")

        try:
            K_Mean = loadModel("src/models/kmeans-clustering.pkl")

            finalDataframePredict["Cluster"] = K_Mean.predict(finalDataframePredict)
            predictedData["Cluster"] = finalDataframePredict["Cluster"]
            
        except Exception as e:
            autolog(f"An error occured {e}")

        # predition started
        autolog("Prediction started")
        clusterID = finalDataframePredict['Cluster'].unique()

        for id in clusterID:
            autolog(f"Training started for Cluster {id}.")
            ## Separating data based on cluster
            clusterDataPredict = finalDataframePredict[finalDataframePredict['Cluster'] == id]
            
            # ## Prepare the feature and Label columns
            x = clusterDataPredict.drop(['Cluster'], axis=1)
        
            # ## Load the model    
        
            pred_model = loadModel(f"{pre.predModelDirs}/{id}.pkl")
            predictedData.loc[predictedData['Cluster'] == id, 'class'] = pred_model.predict(x)
            

        decoder = loadModel("src/models/encoder.pkl")

        predictedData["class"] = decoder.inverse_transform(predictedData["class"].astype(int))
        predictedData.drop(['Cluster'], axis=1, inplace=True)

        autolog("Saving predicted data...")
        pre.exportCsv(predictedData, pre.preprocessedPredictCsv)
        autolog("Data saved")
