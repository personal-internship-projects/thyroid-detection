import src.File_Type_Validation as fv
import src.Data_Validation as dv
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
f = zz.addquotestostring(dic)