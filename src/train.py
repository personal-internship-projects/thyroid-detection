from numbers import Number
import src.File_Type_Validation as fv
import src.Data_Validation as dv
from src.clustering import Kmeansclustering
import src.database_operations as dboc
from src.logger.auto_logger import autolog
import src.preprocesing as prp

x = fv.File_Type_Validation("./src/dataset")
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
db.fetch(db.combinedTrain, "train",  db.schemaPath)


pre = prp.Preprocessing()
pre.createPreprocessedDirectory()
pre.readCsv(pre.trainCsv)
pre.dropUnnecessaryColumns()
pre.replaceWithNan()
pre.encodingCategoricalColumnsTraining()
pre.applyingLogTransformation()
isNullPresent = pre.isnullPresent(pre.dataframe,pre.preprocessedNullCsv)

if (isNullPresent):
    pre.dataframe = pre.imputeNanvalues(pre.dataframe)

X_train, Y_train = pre.seperateLabelfeature('class')
X_train, Y_train = pre.resampleData(X_train, Y_train)

autolog("Clustering started.")
K_Mean = Kmeansclustering() 
numberOfClusters = K_Mean.elbowplot(X_train)

X_train_clusters = K_Mean.create_clusters(X_train, numberOfClusters)
autolog(f"number of clusters are: {numberOfClusters}")

dfTransport = X_train_clusters.join(Y_train)
autolog("Starting export ...")
pre.exportCsv(dfTransport, pre.preprocessedTrainCsv)
autolog("Done.")

print(numberOfClusters)