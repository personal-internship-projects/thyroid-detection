from os import read
from pandas.io.parsers import read_csv
from sklearn import cluster
import src.File_Type_Validation as fv
import src.Data_Validation as dv
from src.clustering import Kmeansclustering
from sklearn.model_selection import train_test_split
import src.database_operations as dboc
from src.logger.auto_logger import autolog
from src.model_operations import saveModel
import src.preprocesing as prp
from src import model_finder as mf
import warnings
warnings.filterwarnings("ignore")


"""x = fv.File_Type_Validation("./src/dataset")
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
db.fetch(db.combinedTrain, "train",  db.schemaPath)"""


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

X_train_clusters = K_Mean.create_clusters(X_train, numberOfClusters)
autolog(f"number of clusters are: {numberOfClusters}")

dfTransport = X_train_clusters.join(Y_train)
autolog("Starting export ...")
pre.exportCsv(dfTransport, pre.preprocessedTrainCsv)
autolog("Done.")

print(numberOfClusters)
from src import test as t

## Training started

finalDataframeTrain = read_csv(f"{pre.preprocessedTrainCsv}/preprocessed.csv")
finalDataframeTest = read_csv(f"{pre.preprocessedTestCsv}/preprocessed.csv")

clusterID = finalDataframeTrain['Cluster'].unique()

for id in clusterID:
    autolog(f"Training started for Cluster {id}.")
    ## Separating data based on cluster
    clusterDataTrain     = finalDataframeTrain[finalDataframeTrain['Cluster'] == id]
    #clusterDataTest = finalDataframeTest[finalDataframeTest['Cluster'] == id]
    
    # ## Prepare the feature and Label columns
    x = clusterDataTrain.drop(['class', 'Cluster'], axis=1)
    y = clusterDataTrain['class']

    # clusterFeatureTest = clusterDataTest.drop(['class', 'Cluster'], axis=1)
    # clusterLabelTest = clusterDataTest['class']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
    
    model = mf.ModelFinder(x_train,y_train,x_test,y_test)
    model.getBestparamsDecisionTree()
    model.getBestparamsRandomForest()
    model.getBestparamsKNN()
    modelName, predModel = model.getBestModel()
    saveModel(f"{pre.modelsDirs}/{modelName}_{id}.pkl", predModel)
    autolog(f"Training for cluster {id} completed successfully")
    