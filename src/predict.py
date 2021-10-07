import src.File_Type_Validation as fv
import src.Data_Validation as dv
import src.database_operations as dboc 

zz =dv.DataValidation()
zz.makeFinalCsvDirectory()
z,g,dic = zz.verifyingSchema()
a= zz.validateColumnLength(g)
b = zz.validateMissingValuesInWholeColumn()
d= zz.getColumnName()
e=zz.addColumnNames(d[:-1], zz.predictCsvPath)
f = zz.addQuotesToString(dic)
db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory()
#db.deleteTable('test')
db.createTable('predict')
db.insertValidatedData(db.finalCsvPredict, "predict")
db.fetchTable(db.combinedPredict, "predict")
