import src.database_operations as dboc 

db = dboc.CassandraOperations()
db.databaseConnection()
db.createPreprocessedCsvDirectory()
db.deleteTable("test12")
db.createTable('test12')
db.insertValidatedData(db.finalCsvTest, "test12")
db.fetch(db.combinedTest,'test12')

