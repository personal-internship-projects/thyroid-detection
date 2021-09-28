import src.File_Type_Validation as fv
import src.Data_Validation as dv
import src.database_operations as dboc 
x = fv.File_Type_Validation("./src/dataset")
x.createCsvDir()
x.convertToCsv()
zz =dv.Data_Validation()
zz.makeFinalCsvDirectory()
z,g,dic = zz.verifyingSchema()
a= zz.validateColumnLength(g)
b = zz.validateMissingValuesInWholeColumn()
d= zz.getColumnName()
e=zz.addColumnNames(d)
f = zz.addQuotesToString(dic)
db = dboc.CassandraOperations()
db.databaseConnection()
db.createTable('ineuron','test')