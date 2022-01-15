import flask
from flask import render_template, request
from os.path import isfile
from src.actions import MLoperations 
from src.logger.auto_logger import autolog
import requests 
import shutil

mlops = MLoperations()
#flask application object
app = flask.Flask(__name__)

@app.route('/', methods=["GET"])
def home_page():
    return render_template('index.html')


@app.route('/test', methods=["POST"])
def start_test():
    mlops.test_model()

def upload_data(data):

    autolog("inside upload_data function")
    global count
    url = "https://api.anonfiles.com/upload"
    header = {'Content-type': 'application/octet-stream'}
    files = {
        'file':("src/dataset/preprocessed/predict/preprocessed.csv", open("src/dataset/preprocessed/predict/preprocessed.csv", "rb"))
    }
    try:
        tmp = requests.post('https://api.anonfiles.com/upload', files=files)
        tmp = tmp.json()
        autolog(f"tmp: {tmp}")
        url = tmp["data"]["file"]["url"]["full"]
        autolog("Uploaded data successfully")
        return 0, f"Uploaded data successfully. <a href={url}>Download link</a>"
    except Exception as e:
        autolog(f"Failed to upload data: {e}")
        return 1, "Failed to upload data"
    finally:
        autolog("closed file")        


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
            data = request.files['file']
            # convert the data to a dictionary
            data.save("src/dataset/csv_operation/PredictCSV/datasetpredict.csv")
            # convert the dictionary to a string
            mlops.prediction()
            code, msg = upload_data("datasetpredict")
            shutil.rmtree("src/dataset/csv_operation/PredictCSV")
            return msg
           


app.run(debug=True)