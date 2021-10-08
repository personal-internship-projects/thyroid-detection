from enum import auto
import cassandra
from cassandra.cluster import Cluster, QueryExhausted, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.metadata import Token
from src.logger.auto_logger import autolog
from os import listdir, makedirs
from os.path import isdir
import csv
import pandas as pd
import json
import configparser
from shutil import rmtree

#from src.ttp import pandas_factory
class CassandraOperations:
    def __init__(self):
        self.dbpath                =  "src/DATABASE_OPERATIONS"
        self.schemaPath            =  "src/schema_training.json" 
        self.schemaPathPredict     =  "src/schema_predict.json"
        self.finalCsvTest          =  "src/dataset/final_csv/test"
        self.finalCsvTrain         =  "src/dataset/final_csv/train"
        self.finalCsvPredict       =  "src/dataset/final_csv/predict"
        self.goodCsvPath           =  "src/dataset/csv_operation/GoodCSV"
        self.badCsvPath            =  "src/dataset/csv_operation/BadCSV"
        self.combinedTrain         =  "src/dataset/combined_csv/train"
        self.combinedTest          =  "src/dataset/combined_csv/test"
        self.combinedPredict       =  "src/dataset/combined_csv/predict"
        self.keyspace_name         =  "ineuron"

    def createPreprocessedCsvDirectory(self, combinedDirectoryLocation):
        autolog("Creating Directory for combined csv ...")
        if not isdir(combinedDirectoryLocation):
            makedirs(combinedDirectoryLocation)  
        autolog("Directories created.")


    @staticmethod
    def schemaParser(schemaPath) -> dict():
        dic = dict()
        try:
            with open(schemaPath, 'r') as f:
                dic = json.load(f)
            
                f.close()
        except Exception as e:
            autolog("Error: " + str(e))
            raise e
        return dic["ColName"]


    def columnNameRetriever(self) -> list():
        scheme_dict = self.schemaParser(self.schema_path)
        lst_dict = [x for x in scheme_dict]
        lst = ' '.join([str(elem)+"," for elem in lst_dict])
        return lst

    def databaseConnection(self):
        try:
            autolog("Trying to initialize Connection...")
            obj = configparser.ConfigParser()
            obj.read("./config.ini")
            clientId   = obj["DATABASE_CREDS"]["client_id"]
            secret     = obj["DATABASE_CREDS"]["client_secret"]
            bundlePath = obj["DATABASE_CREDS"]["secure_bundle_path"]
            
            cloud_config= {
                'secure_connect_bundle': "src/secure-connect-test.zip" #bundle
            }
            #client_id = "sBKfIXPONuTQnWvElwjckcyG"
            #client_secret = "jLDP1mfaC7ZNjZZMxcLniG14X-oQ0kBMJNU,95tkLtm5Nhmqaa8a8ztgmRXxtaq+MZhtLowgNns7KdH_g7cDy0xrb1WfADqNNf5c35BxYPWs47TZYgJtuWele1TtJT3s"

            auth_provider = PlainTextAuthProvider("zHTNaJwOaGDlFozNJEabKnoZ", "s4XthkziU7Av8QLU+YI_MHXYZIds_uR7tphlkkFe1XDB+r+k8DzmG8G+rqbPCXZ2_QLc,NtZEjYKnSHMeAz9C8W,WJ-MapEYZ,j9kufqSU_MYQGt8KP5K2lQPJp-w8Kx")
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


    def createTable(self, table_name, schemaPath):
        autolog("Function Started")
        column_dict = self.schemaParser(schemaPath)
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


    def columnNameRetriever(self, schemaPath) -> list():
        scheme_dict = self.schemaParser(schemaPath)
        lst_dict = [x for x in scheme_dict]
        lst = ' '.join([str(elem)+"," for elem in lst_dict])
        return lst[:-1]


    def insertValidatedData(self, path, table_name, schemaPath):
        lst = self.columnNameRetriever(schemaPath)
        try:
            count = self.session.execute(f"SELECT COUNT(*) FROM {self.keyspace_name}.{table_name};")
            try:
                count = count[0].count
            except :
                count = 1
            
            for files in listdir(path):
                autolog(f"Opening {files}...")
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
                            else:
                                query = f"INSERT INTO {self.keyspace_name}.{table_name} (id,{lst}) VALUES ({count},{words});"
                                BATCH_STMT += query
                                BATCH_STMT += ' APPLY BATCH;'
                                self.session.execute(BATCH_STMT)
                                BATCH_STMT = "BEGIN BATCH "
                                batch_size = 0
                            count += 1
                    if batch_size != 0:
                        BATCH_STMT += ' APPLY BATCH;'
                        self.session.execute(BATCH_STMT)
                        print(BATCH_STMT)
                        batch_size = 0
            autolog(f"Inserted data successfully into database.") 
            self.deleteCsv()            
        except Exception as e:
            autolog("Failed to insert data" ,3)  


    def pandas_factory(self,colnames, rows):
        return pd.DataFrame(rows, columns=colnames)
            
    def fetch(self, path, tableName, schemaPath):
        lst = self.columnNameRetriever(schemaPath)

        autolog("Fetching data from database...")
        query = (f"SELECT id,{lst} FROM {self.keyspace_name}.{tableName};")
        self.session.row_factory = self.pandas_factory
        self.session.default_fetch_size = 1000000
        try:
            rows = self.session.execute(query)
            autolog("Query executed successfully.")
        except Exception as e:
            autolog(f"Query failed to execute. {e}", 3)
            exit(-1)

        df = pd.DataFrame()
        df = df.append(rows._current_rows)
        while rows.has_more_pages == True:
            rows.fetch_next_page()
            df = df.append(rows._current_rows)
        df = df.round(4)

        if tableName == "test":
            df.iloc[:,-1].to_csv(f"{path}/class.csv", index=None, header=True)
            df.iloc[:,:-1].to_csv(f"{path}/combined.csv", index=None, header=True)
        else:
            df.to_csv(f"{path}/combined.csv", index=None, header=True)
        autolog(f"Stored fetched data to {path}/combined.csv")
            

