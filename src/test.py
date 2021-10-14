import src.database_operations as dboc
import src.preprocesing as prp

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
pre.mappingCategoricalColumns()
pre.getDummies()
pre.labelEncoding()
pre.exportCsv(pre.testCsv)
