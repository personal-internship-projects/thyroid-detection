import os
import shutil
import src.File_Type_Validation as fv
import json
from src.logger.auto_logger import autolog
from pandas import read_csv, DataFrame
import re
import src.database_operations as db


class DataValidation:
    def __init__(self) -> None:
        self.schemaPath = "src/schema_training.json"
        self.schemaPathPredict = "src/schema_predict.json"
        self.finalCsv = "src/dataset/final_csv"
        self.finalCsvTest = "src/dataset/final_csv/test"
        self.finalCsvTrain = "src/dataset/final_csv/train"
        self.finalCsvPredict = "src/dataset/final_csv/predict"
        self.goodCsvPath = "src/dataset/csv_operation/GoodCSV"
        self.badCsvPath = "src/dataset/csv_operation/BadCSV"
        self.predictCsvPath = "src/dataset/csv_operation/PredictCSV"

    def makeFinalCsvDirectory(self, finalDirectoryLocation):
        if not os.path.isdir(finalDirectoryLocation):
            os.makedirs(finalDirectoryLocation)

    def verifyingSchema(self, schemaLoc):
        """This fuction is used to verify the schema of the csv file


        Args:
            schemaLoc ([json]): [storing the schema of the csv file]

        Raises:
            ValueError: [Raise error if the schema is not matching]
            KeyError: [KeyError if key is not found in the schema]
            e: [If any other error occurs]

        Returns:
            [Column_Names]: [Name of the columns of the csv file]
            [Number_Of_Columns]: [Number of columns in the csv file]
            [dic]: [dictionary of the schema]
        """
        try:
            with open(schemaLoc, 'r') as f:
                dic = json.load(f)
                f.close()

            column_names = dic['ColName']
            NumberOfColumns = dic['NumberOfColumns']

        except ValueError:
            autolog("ValueError:Value not found inside schema_training.json")
            raise ValueError

        except KeyError:
            autolog("KeyError:Key not found inside schema_training.json")
            raise KeyError

        except Exception as e:
            autolog("Error: " + e)
            raise e
        return column_names, NumberOfColumns, dic

    def validateColumnLength(self, NumberOfColumns, path):
        """This function is used to validate the length of the column in the csv file

        Args:
            NumberOfColumns ([int]): [Storing the number of columns in the csv file]
            path ([string]): [Storing the path of the csv file]

        Raises:
            OSError: [if error occurs while moving the file]
            e: [if any other error occurs]
        """
        try:
            autolog("Column Length Validation Started!!")
            for files in os.listdir(path):
                csv = read_csv(f"{path}/{files}")
                if csv.shape[1] == NumberOfColumns:
                    pass
                else:
                    shutil.copy(path + "/" + files, self.badCsvPath)

        except OSError:
            autolog(f"Error Occured while moving the file :: {OSError}")
            raise OSError
        except Exception as e:
            autolog(f"Error Occured:: {e}")
            raise e

    def validateMissingValuesInWholeColumn(self, path):
        """This function is used to validate the missing values in the whole column

        Args:
            path ([string]): [storing the path of the csv file]
        """
        try:
            autolog("Missing Values Validation Started!!")

            for files in os.listdir(path):
                csv = read_csv(path + files)
                count = 0
                for cols in csv:
                    if len(csv[cols]) - csv[cols].count == len(csv[cols]):
                        shutil.copy(path + files, self.badCsvPath)
                        count += 1
                        break
        except:
            pass

    def getColumnName(self, schemaPath):
        """This function is used to get the column name from the schema file

        Args:
            schemaPath ([string]): [Storing the path of the schema file]

        Returns:
            [list]: [Storing the column name]
        """
        lst = list(db.CassandraOperations.schemaParser(schemaPath).keys())
        return lst

    def addColumnNames(self, lst, path):
        """This function is used to add the column name in the csv file

        Args:
            lst ([list]): [Contains the column name]
            path ([type]): [Containing the path of the csv file]
        """
        autolog("Adding column named to csv ...")
        for files in os.listdir(path):
            csv = DataFrame()
            df1 = read_csv(f"{path}/{files}")
            count = 0
            for labels in lst:
                csv[f"{labels}"] = df1.iloc(axis=1)[count]
                count += 1

            if "data" in files:
                autolog(f"Adding {files} to train dataset")
                csv.to_csv(f"{self.finalCsvTrain}/{files}",
                           index=None, header=True)
            elif "test" in files:
                autolog(f"Adding {files} to test dataset")
                csv.to_csv(f"{self.finalCsvTest}/{files}",
                           index=None, header=True)
            else:
                autolog(f"Adding {files} to predict dataset")
                csv.to_csv(f"{self.finalCsvPredict}/{files}",
                           index=None, header=True)

        autolog("Done.")

    def addQuotesToString(self, dict):
        """This function is used to add quotes to the string

        Args:
            dict ([dictionary]): [contains the dictionary of the schema file]
        """
        autolog("Adding quotes to strings in dataset started...")
        for x in os.listdir(self.finalCsv):
            mainDir = f"{self.finalCsv}/{x}"
            for files in os.listdir(mainDir):
                data = read_csv(f"{mainDir}/{files}")
                data.iloc[:, -1] = data.iloc[:, -
                                             1].apply(lambda x: re.match(r'([^.|]*)', x).group(1))

                column = [x for x in dict["ColName"]
                          if dict["ColName"][x] == "varchar"]

                for col in data.columns:
                    if col in column:
                        data[col] = data[col].apply(lambda x: f"'{str(x)}'")
                    elif col not in column:
                        data[col] = data[col].replace('?', "null")

                os.remove(f"{mainDir}/{files}")
                data.to_csv(f"{mainDir}/{files}", index=None, header=True)

                autolog(f"added quotes {files}.csv completed. ")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(args.path)

    zz = DataValidation()
    zz.goodCsvPath = args.path
    zz.makeFinalCsvDirectory()
    z, g, dic = zz.verifyingSchema()
    a = zz.validateColumnLength(g)
    b = zz.validateMissingValuesInWholeColumn()
    d = zz.getColumnName()
    e = zz.addColumnNames(d[:-1])
    f = zz.addQuotesToString(dic)
