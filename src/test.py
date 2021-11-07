import src.database_operations as dboc
from src.model_operations import loadModel
import src.preprocesing as prp
from src.clustering import Kmeansclustering
from src.logger.auto_logger import autolog
"""
db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory(db.combinedTest)
db.deleteTable("test")
db.createTable('test',db.schemaPath)
db.insertValidatedData(db.finalCsvTest, "test", db.schemaPath)
db.fetch(db.combinedTest,'test', db.schemaPath)
"""
pre = prp.Preprocessing()
pre.createPreprocessedDirectory()
pre.readCsv(pre.testCsv)
pre.dropUnnecessaryColumns()
pre.replaceWithNan()
pre.encodingCategoricalColumnsPrediction()
is_null_present = pre.isnullPresent(pre.dataframe, pre.preprocessedNullCsv)

X_test, Y_test = pre.seperateLabelfeature('class')

if (is_null_present):
    X_test = pre.imputeNaNValuesOnTestAndPredict(X_test)

## After resampling data, we are separating X, Y 
## for applying quantile transformer

autolog("Applying Quantile Transformer...")
X_test = pre.quantileTransformer(X_test)
autolog("Quantile Transformer applied")

try:
    K_Mean = loadModel("src/models/kmeans-clustering.pkl") 

    X_test["Cluster"] = K_Mean.predict(X_test)
except Exception as e:
    autolog(f"An error occured {e}")
    
dfTransport = X_test.join(Y_test)
pre.exportCsv(dfTransport, pre.preprocessedTestCsv)



