import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from src.logger.auto_logger import autolog
import json
class CassandraOperations:
    def __init__(self):
        self.dbpath         =  "src/DATABASE_OPERATIONS"
        self.schema_path    =  "src/schema_training.json" 
        self.finalCsvTest   =  "src/dataset/final_csv/test"
        self.finalCsvTrain  =  "src/dataset/final_csv/train"
        self.goodCsvPath    =  "./src/dataset/csv_operation/GoodCSV"
        self.badCsvPath     =  "./src/dataset/csv_operation/BadCSV"

    
    def schemaParser(self):
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
                'secure_connect_bundle': 'G:\\Ineuron\\secure-connect-test.zip'
            }
            auth_provider = PlainTextAuthProvider('djMBOJUicLZEvpHTGZFRxDBI', 'WCYx-3FA+gBijXY.YqKWUbMnLh8Wg2bS5ZPuUU8ex4Hzlh6IhmZZbtT81ZAOxNYy_ld5HhT.D76SfBtSfph6ZMeXWZm50ozHjic2A-Dihriicj2nQcOe0.-,fKt1AfY4')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            self.session = cluster.connect()
            autolog("Connection started sucessfully.")
        except ConnectionError:
            autolog("Error while connecting to database.")
    


    def createTable(self, keyspace_name, table_name):
        autolog("Function Started")
        column_dict = self.schemaParser()
        for column_name in column_dict.keys():
            if column_name=='age':
                continue
            datatype_of_columns = column_dict[column_name]
            try:
                self.session.execute(f'ALTER TABLE {keyspace_name}.{table_name} ADD "{column_name}" {datatype_of_columns}')
            except cassandra.InvalidRequest as e :
                autolog(f"Column {column_name} already exists.")
            except Exception as e:
                autolog("TABLE {table_name} does not exist.")
                try:
                    self.session.execute(f"CREATE TABLE  {keyspace_name}.{table_name} (id int PRIMARY KEY, {column_name} {datatype_of_columns})")
                    autolog(f"TABLE {table_name} created.")
                except :
                    autolog("Failed to create table {table_name}")
                



