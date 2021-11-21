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
f = zz.addQuotesToString(dic)
db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory(db.combinedPredict)
db.deleteTable('predict')
db.createTable('predict', db.schemaPathPredict)
db.insertValidatedData(db.finalCsvPredict, "predict", db.schemaPathPredict)
db.fetch(db.combinedPredict, "predict", db.schemaPathPredict)

pre = prp.Preprocessing()
pre.createPreprocessedDirectory()
pre.readCsv(pre.predictCsv)
pre.dropUnnecessaryColumns()
pre.replaceWithNan()
pre.encodingCategoricalColumnsPrediction()
is_null_present = pre.isnullPresent(
    pre.dataframe, pre.preprocessedNullCsvPredict)

data = pre.dataframe
if (is_null_present):
    data = pre.imputeNaNValuesOnTestAndPredict(data)

autolog("Applying Logarithmic Transformer...")
data = pre.LogTransformer(data)
autolog("Log Transformer applied")

try:
    K_Mean = loadModel("src/models/kmeans-clustering.pkl")

    data["Cluster"] = K_Mean.predict(data)
except Exception as e:
    autolog(f"An error occured {e}")

autolog("Savind data after preprocessing...")
pre.exportCsv(data, pre.preprocessedPredictCsv)
