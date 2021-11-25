import pandas as pd
from pandas.core.frame import DataFrame
import src.Data_Validation as dv
import src.database_operations as dboc
import src.preprocesing as prp
from src.logger.auto_logger import autolog
from src.model_operations import loadModel

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