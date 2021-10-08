import src.Data_Validation as dv
import src.database_operations as dboc 


zz =dv.DataValidation()
zz.makeFinalCsvDirectory(zz.finalCsvPredict)
z,g,dic = zz.verifyingSchema(zz.schemaPathPredict)
a= zz.validateColumnLength(g, zz.predictCsvPath)
b = zz.validateMissingValuesInWholeColumn(zz.predictCsvPath)
d= zz.getColumnName(zz.schemaPathPredict)
e=zz.addColumnNames(d, zz.predictCsvPath)
f = zz.addQuotesToString(dic)
db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory(db.combinedPredict)
db.deleteTable('predict')
db.createTable('predict', db.schemaPathPredict)
db.insertValidatedData(db.finalCsvPredict, "predict", db.schemaPathPredict)
db.fetch(db.combinedPredict, "predict", db.schemaPathPredict)
