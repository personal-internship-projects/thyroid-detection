from pandas.io.parsers import read_csv
from sklearn import cluster
import src.File_Type_Validation as fv
import src.Data_Validation as dv
from src.clustering import Kmeansclustering
import src.database_operations as dboc
from src.logger.auto_logger import autolog
import src.preprocesing as prp
from sklearn.model_selection import train_test_split

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
isNullPresent = pre.isnullPresent(pre.dataframe,pre.preprocessedNullCsv)

if (isNullPresent):
    pre.imputeNanvalues(pre.dataframe)

X_train, Y_train = pre.seperateLabelfeature('class')
X_train, Y_train = pre.resampleData(pre.preprocessedTrainCsv, X_train, Y_train)

## After resampling data, we are separating X, Y 
## for applying quantile transformer

autolog("Applying Quantile Transformer...")
X_train = pre.applyStandardScaler(X_train)
autolog("Quantile Transformer applied")

tmp_dataframe = X_train.join(Y_train)

## After applying data transformation on numeric column 
## this step will remove outliers from numeric columns

pre.removeOutlier(tmp_dataframe)
print(tmp_dataframe['class'].unique())

## After removing outlier from whole dataset, we are separating X and Y again
## This step is done for clustering

X_train, Y_train = pre.seperateLabelfeature('class')

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

## Training started

autolog("Training started.")
finalDataframe = read_csv(f"{pre.preprocessedTrainCsv}/preprocessed.csv")
clusterID = finalDataframe['Cluster'].unique()

for id in clusterID:
    
    ## Separating data based on cluster
    clusterData = finalDataframe[finalDataframe['Cluster'] == id]
    
    ## Prepare the feature and Label columns
    ## clusterFeatuer = X
    ## clusterLabel = Y
    clusterFeature = clusterData.drop(columns=['class', 'Cluster'])
    clusterLabel = clusterData['class']
    
    X_train, X_test, Y_train, Y_test = train_test_split(clusterFeature, clusterLabel, test_size=0.2, random_state=500, stratify=clusterLabel)
    import numpy as np
    np.random.seed(42)
    print(f"Y_Train = {np.unique(Y_train, return_counts=True)}")
    print(f"Y_Test = {np.unique(Y_test, return_counts=True)}", end="\n\n")


    