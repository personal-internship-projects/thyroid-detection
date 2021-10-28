import src.database_operations as dboc
from src.model_operations import loadModel
import src.preprocesing as prp
from src.clustering import Kmeansclustering
from src.logger.auto_logger import autolog

db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory(db.combinedTest)
db.deleteTable("test")
db.createTable('test',db.schemaPath)
db.insertValidatedData(db.finalCsvTest, "test", db.schemaPath)
db.fetch(db.combinedTest,'test', db.schemaPath)

pre = prp.Preprocessing()
pre.createPreprocessedDirectory()
pre.readCsv(pre.testCsv)
pre.dropUnnecessaryColumns()
pre.replaceWithNan()
pre.encodingCategoricalColumnsPrediction()

is_null_present = prp.isnullPresent(prp.dataframe,prp.preprocessedNullCsv)

if (is_null_present):
    prp.dataframe = prp.imputeNanvalues(prp.dataframe)

X_test, Y_test = pre.seperateLabelfeature('class')

K_Mean = loadModel("src/models/kmeans-clustering.pkl") 
numberOfClusters = K_Mean.elbowplot(X_test)
X_train_clusters = K_Mean.create_clusters(X_test, numberOfClusters)
autolog(f"number of clusters are: {numberOfClusters}")

dfTransport = X_train_clusters.join(Y_test)
autolog("Starting export ...")
pre.exportCsv(dfTransport, pre.preprocessedTrainCsv)




