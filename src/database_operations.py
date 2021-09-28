import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from src.logger.auto_logger import autolog
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
            autolog("Error: " + e)
            raise e
        return dic


    def databaseconnection(self):
        try:
            autolog("Trying to initialize Connection...")
            cloud_config= {
                'secure_connect_bundle': 'G:\\Ineuron\\secure-connect-test.zip'
            }
            auth_provider = PlainTextAuthProvider('djMBOJUicLZEvpHTGZFRxDBI', 'WCYx-3FA+gBijXY.YqKWUbMnLh8Wg2bS5ZPuUU8ex4Hzlh6IhmZZbtT81ZAOxNYy_ld5HhT.D76SfBtSfph6ZMeXWZm50ozHjic2A-Dihriicj2nQcOe0.-,fKt1AfY4')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect()
            autolog("Connection started sucessfully.")
        except ConnectionError:
            autolog("Error while connecting to database.")
        return session
    


    def create_table(self,keyspace_name):
        


