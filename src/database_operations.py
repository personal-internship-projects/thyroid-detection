import cassandra
from cassandra.cluster import Cluster, QueryExhausted
from cassandra.auth import PlainTextAuthProvider
from src.logger.auto_logger import autolog
from os import listdir
import csv
import json
class CassandraOperations:
    def __init__(self):
        self.dbpath         =  "src/DATABASE_OPERATIONS"
        self.schema_path    =  "src/schema_training.json" 
        self.finalCsvTest   =  "src/dataset/final_csv/test"
        self.finalCsvTrain  =  "src/dataset/final_csv/train"
        self.goodCsvPath    =  "./src/dataset/csv_operation/GoodCSV"
        self.badCsvPath     =  "./src/dataset/csv_operation/BadCSV"
        self.keyspace_name  =  "ineuron"

    
    def schemaParser(self) -> dict():
        dic = dict()
        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
            
                f.close()
        except Exception as e:
            autolog("Error: " + str(e))
            raise e
        return dic["ColName"]


    def databaseConnection(self):
        try:
            autolog("Trying to initialize Connection...")
            cloud_config= {
                'secure_connect_bundle': 'src/secure-connect-test.zip'
            }
            auth_provider = PlainTextAuthProvider('djMBOJUicLZEvpHTGZFRxDBI', 'WCYx-3FA+gBijXY.YqKWUbMnLh8Wg2bS5ZPuUU8ex4Hzlh6IhmZZbtT81ZAOxNYy_ld5HhT.D76SfBtSfph6ZMeXWZm50ozHjic2A-Dihriicj2nQcOe0.-,fKt1AfY4')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            self.session = cluster.connect()
            autolog("Connection started sucessfully.")
        except ConnectionError:
            autolog("Error while connecting to database.")
    
    def deleteTable (self, table_name):
        try:
            self.session.execute(f"Drop table if exists {self.keyspace_name}.{table_name}")
            autolog(f"Deleted {table_name} successfully")
        except:
            autolog(f"Failed to delete table {table_name}")

    def createTable(self, table_name):
        autolog("Function Started")
        column_dict = self.schemaParser()
        for column_name in column_dict.keys():
            datatype_of_columns = column_dict[column_name]
            try:
                self.session.execute(f'ALTER TABLE {self.keyspace_name}.{table_name} ADD "{column_name}" {datatype_of_columns}')
                autolog(f"Adding column {column_name}")
            except Exception as e:
                autolog(f"Column {column_name} already exists or TABLE {table_name} does not exist.")
                try:
                    self.session.execute(f"CREATE TABLE  {self.keyspace_name}.{table_name} (id int PRIMARY KEY, {column_name} {datatype_of_columns})")
                    autolog(f"TABLE {table_name} created.")
                except :
                    autolog(f"Failed to create table {table_name}")


    def insertValidatedData(self, path, table_name):
        scheme_dict = self.schemaParser()
        lst_dict = [x for x in scheme_dict]
        lst = ' '.join([str(elem)+"," for elem in lst_dict])
        lst = lst[:-1]
        count = 1

        for files in listdir(path):
            with open (f"{path}/{files}", "r") as csv_file:
                next(csv_file)
                reader = csv.reader(csv_file, delimiter="\n")
                BATCH_STMT = 'BEGIN BATCH '
                batch_size = 0
                for rows in enumerate(reader):
                    for words in rows[1]:
                        
                        if batch_size <= 500:
                            query = f"INSERT INTO {self.keyspace_name}.{table_name} (id,{lst}) VALUES ({count},{words});"
                            BATCH_STMT += query
                            batch_size += 1
                            count += 1
                        else:
                            BATCH_STMT += ' APPLY BATCH;'
                            self.session.execute(BATCH_STMT)
                            print(BATCH_STMT)
                            BATCH_STMT = "BEGIN BATCH "
                            batch_size = 0
                        
                if batch_size != 0:
                    BATCH_STMT += ' APPLY BATCH;'
                    self.session.execute(BATCH_STMT)
                    print(BATCH_STMT)
                    batch_size = 0 


    
                



