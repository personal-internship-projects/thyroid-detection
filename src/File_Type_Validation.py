import os
import shutil
from src.logger.auto_logger import autolog

class File_Type_Validation:
    def __init__(self, path) -> None:
        self.path = path
        self.goodCsvPath    = self.path+"/csv_operation/GoodCSV"
        self.badCsvPath     = self.path+"/csv_operation/BadCSV"
        self.predictCsvPath = self.path+"/csv_operation/PredictCSV"

    def createCsvDir (self):

        autolog("Creating Directory for csv operations ...")
        if not os.path.isdir(self.goodCsvPath):
            os.makedirs(self.goodCsvPath)  
        if not os.path.isdir(self.badCsvPath):
            os.makedirs(self.badCsvPath)
        if not os.path.isdir(self.predictCsvPath):
            os.makedirs(self.predictCsvPath)
            
        autolog("Directories created.")


    def convertToCsv(self):

        autolog("Renaming required files to csv ...")
        file_extensions = (".data",".test")
        raw_files = f"{self.path}/batches_of_data"
        

        for i in os.listdir(path=raw_files):
            if i.endswith(file_extensions):
                new_name = f"{self.goodCsvPath}/{i}.csv"
                shutil.copy(f"{raw_files}/{i}", self.goodCsvPath)
                shutil.move(f"{self.goodCsvPath}/{i}", new_name)
                autolog(f"Copying and renaming {i} to {self.goodCsvPath} ...")

            else:
                shutil.copy(f"{raw_files}/{i}", self.badCsvPath)
                autolog(f"Copying bad file {i} to {self.badCsvPath}")


if __name__=='__main__':
    x = File_Type_Validation("./src/dataset")
    x.createCsvDir()
    x.convertToCsv()