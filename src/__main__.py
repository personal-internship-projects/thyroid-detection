import src.File_Type_Validation as fv
import src.Data_Validation as dv
import src.database_operations as dboc 
x = fv.File_Type_Validation("./src/dataset")
"""x.createCsvDir()
x.convertToCsv()
zz =dv.DataValidation()
zz.makeFinalCsvDirectory()
z,g,dic = zz.verifyingSchema()
a= zz.validateColumnLength(g)
b = zz.validateMissingValuesInWholeColumn()
d= zz.getColumnName()
e=zz.addColumnNames(d)
f = zz.addQuotesToString(dic)"""
db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory()
db.deleteTable('test')
db.createTable('test')
db.insertValidatedData(db.finalCsvTrain, "test")
db.fetch(db.combinedTrain)