from flask import Flask, render_template,request
from src.utils.all_utils import read_yaml
from flask_cors import CORS, cross_origin
import webbrowser
from threading import Timer
import pandas as pd
import pickle
import logging
import os
import subprocess

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'Prediction.log'), level=logging.INFO, format=logging_str,
                    filemode="a")


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('home.html')

@app.route('/train', methods=['POST', 'GET'])
@cross_origin()
def train_function():

    """
    This function is used to train the model.
    :param config_path:
    :return:
    """
    config_path = "config/config.yaml"
    config = read_yaml(config_path)
    artifacts_dir = config['artifacts']['artifacts_dir']
    model_dir = config["artifacts"]["model_dir"]

    # Random Forest
    model_filename_RF = config["artifacts"]["model_file_random_forest"]
    model_path_RF = os.path.join(artifacts_dir, model_dir, model_filename_RF)

    # XGBoost Path

    model_filename_XG = config["artifacts"]["model_file_xgboost"]
    model_path_XG = os.path.join(artifacts_dir, model_dir, model_filename_XG)


    if request.method == 'POST':
        subprocess.run(["dvc", "repro"])
        
        try:
            # Inputs based to data
            GENDER = request.form['GENDER']
            if GENDER == 'Male':
                GENDER = 1
            else:
                GENDER = 2
            MARRIED = request.form['MARRIED']
            if MARRIED == 'Yes':
                MARRIED = 1
            else:
                MARRIED = 2
            DEPENDENT = int(request.form['DEPENDENT'])
            if DEPENDENT > 3:
                DEPENDENT = 3
            else:
                DEPENDENT = DEPENDENT
            EDUCATION  = request.form['EDUCATION']
            if EDUCATION == 'Graduate':
                EDUCATION = 1
            else:
                EDUCATION = 2
            SELF_EMPLOYED =  request.form['SELF_EMPLOYED']
            if SELF_EMPLOYED == 'Yes':
                SELF_EMPLOYED = 2
            else:
                SELF_EMPLOYED = 1
            APPLICANT_INCOME =  int(request.form['APPLICANT_INCOME'])
            COAPPLICANT_INCOME = int(request.form['COAPPLICANT_INCOME'])
            LOAN_AMOUNT = float(request.form['LOAN_AMOUNT'])
            LOAN_AMOUNT_TERM = float(request.form['LOAN_AMOUNT_TERM'])
            CREDIT_HISTORY = float(request.form['CREDIT_HISTORY'])
            PROPERTY_AREA = request.form['PROPERTY_AREA']
            if PROPERTY_AREA == 'Urban':
                PROPERTY_AREA = 1
            elif PROPERTY_AREA == 'Semiurban':
                PROPERTY_AREA = 2
            else:
                PROPERTY_AREA = 3
            MODEL_TYPE = request.form['MODEL_TYPE']
            if MODEL_TYPE == 'RandomForest':
                model = pickle.load(open(model_path_RF, "rb"))
            else:
                model = pickle.load(open(model_path_XG, "rb"))
            
            # Prediction
            prediction = model.predict([[
                GENDER,
                MARRIED,
                DEPENDENT,
                EDUCATION,
                SELF_EMPLOYED,
                APPLICANT_INCOME,
                COAPPLICANT_INCOME,
                LOAN_AMOUNT,
                LOAN_AMOUNT_TERM,
                CREDIT_HISTORY,
                PROPERTY_AREA
            ]])
            output = round(prediction[0], 2)

            if output == 1:
                text = 'Approved'
            else:
                text = 'Rejected'

            return render_template('home.html', prediction_text=f"Your Loan Status is {text}")
 
        except Exception as e:
            logging.error("Error in giving Input!!!!")
            logging.error(e)
    else:
        return render_template('home.html')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8000/')


def start_app():
    Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=8000,debug=True)



if __name__ == "__main__":
    try:
        logging.info(">>>>>>Starting Stage 5 Prediction !!!")
        start_app()
        logging.info(">>>>>>Stage 5 Prediction Completed !!!")
    except Exception as e:
        logging.error(">>>>>>Error in Stage 5 Prediction !!!")
        logging.error(e)