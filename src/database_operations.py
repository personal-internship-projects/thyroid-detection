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

#from src.ttp import pandas_factory
class CassandraOperations:
    def __init__(self):
        self.dbpath                =  "src/DATABASE_OPERATIONS"
        self.schema_path           =  "src/schema_training.json" 
        self.finalCsvTest          =  "src/dataset/final_csv/test"
        self.finalCsvTrain         =  "src/dataset/final_csv/train"
        self.goodCsvPath           =  "./src/dataset/csv_operation/GoodCSV"
        self.badCsvPath            =  "./src/dataset/csv_operation/BadCSV"
        self.combinedTrain         =  "./src/dataset/combined_csv/train"
        self.combinedTest          =  "./src/dataset/combined_csv/test"
        self.keyspace_name         =  "ineuron"

    def createPreprocessedCsvDirectory(self):
        autolog("Creating Directory for combined csv ...")
        if not isdir(self.combinedTest):
            makedirs(self.combinedTest)  
        if not isdir(self.combinedTrain):
            makedirs(self.combinedTrain)
        autolog("Directories created.")



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
            obj = configparser.ConfigParser()
            obj.read("./config.ini")
            clientId   = obj["DATABASE_CREDS"]["client_id"]
            secret     = obj["DATABASE_CREDS"]["client_secret"]
            bundlePath = obj["DATABASE_CREDS"]["secure_bundle_path"]
            
            cloud_config= {
                'secure_connect_bundle': bundlePath
            }
            #client_id = "djMBOJUicLZEvpHTGZFRxDBI"
            #client_secret = "WCYx-3FA+gBijXY.YqKWUbMnLh8Wg2bS5ZPuUU8ex4Hzlh6IhmZZbtT81ZAOxNYy_ld5HhT.D76SfBtSfph6ZMeXWZm50ozHjic2A-Dihriicj2nQcOe0.-,fKt1AfY4"

            auth_provider = PlainTextAuthProvider(clientId, secret)
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
        self.lst = lst
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
        except Exception as e:
            autolog("Failed to insert data" ,3)  


    def pandas_factory(self,colnames, rows):
        return pd.DataFrame(rows, columns=colnames)
            
    def fetch(self, path):
        scheme_dict = self.schemaParser()
        lst_dict = [x for x in scheme_dict]
        lst = ' '.join([str(elem)+"," for elem in lst_dict])
        lst = lst[:-1]
        autolog("Fetching data from database...")
        query = (f"SELECT id,{self.lst} FROM {self.keyspace_name}.test;")
        self.session.row_factory = self.pandas_factory
        self.session.default_fetch_size = 1000000
        try:
            rows = self.session.execute(query)
            autolog("Query executed successfully.")
        except Exception as e:
            autolog("Query failed to execute.", 3)
            exit(-1)

        df = pd.DataFrame()
        df = df.append(rows._current_rows)
        while rows.has_more_pages == True:
            rows.fetch_next_page()
            df = df.append(rows._current_rows)
        df = df.round(4)
        df.to_csv(f"{path}/combined.csv", index=None, header=True)
        autolog(f"Stored fetched data to {path}/combined.csv")
            
