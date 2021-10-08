import src.database_operations as dboc 

db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory(db.combinedTest)
db.deleteTable("test")
db.createTable('test')

db.insertValidatedData(db.finalCsvTest, "test", db.schemaPath)
db.fetch(db.combinedTest,'test', db.schemaPath)

