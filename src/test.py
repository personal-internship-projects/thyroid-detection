import src.database_operations as dboc
from src.model_operations import loadModel
import src.preprocesing as prp
from src.clustering import Kmeansclustering
from src.logger.auto_logger import autolog

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

