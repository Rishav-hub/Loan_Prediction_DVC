from numpy.lib.shape_base import split
from src.utils.all_utils import read_yaml, create_directory, save_local_df, save_reports
import argparse
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pickle
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'Evaluation.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def evaluation_matrices(actual_values, predicted_values):
    """
    This function calculates the evaluation matrices for the given actual and predicted values.
    :param actual_values:
    :param predicted_values:
    :return:
    """
    # Calculate the evaluation matrices
    accuracy = accuracy_score(actual_values, predicted_values)
    f1 = f1_score(actual_values, predicted_values)
    roc_auc = roc_auc_score(actual_values, predicted_values)
    return accuracy, f1, roc_auc

def evaluate(config_path, params_path):
    """
    This function evaluates the model on the test set.
    :param config_path:
    :param params_path:
    :return:
    """


    config = read_yaml(config_path)
    params = read_yaml(params_path)
    models_list= []

    # Get the data
    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir= config['artifacts']['split_data_dir']
    test_data_filename= config['artifacts']['test']

    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)

    # read the test data
    test_data = pd.read_csv(test_data_path)
    test_y = test_data["Loan_Status"]
    test_x = test_data.drop("Loan_Status", axis=1)

    # Model Directory

    model_dir = config["artifacts"]["model_dir"]

    # RandomForest Path
    model_filename_RF = config["artifacts"]["model_file_random_forest"]
    model_path_RF = os.path.join(artifacts_dir, model_dir, model_filename_RF)
    try:
        logging.info("Started Loading the Random Forest Model!!!!")
        RandomForest_model = pickle.load(open(model_path_RF, 'rb'))
        models_list.append(RandomForest_model)
        logging.info("Finished Loading the Random Forest Model!!!!")
    except Exception as e:
        logging.error("Error in Loading the Random Forest Model!!!!")
        logging.error(e)

    # XGBoost Path
    model_filename_XG = config["artifacts"]["model_file_xgboost"]
    model_path_XG = os.path.join(artifacts_dir, model_dir, model_filename_XG)

    try:
        logging.info("Started Loading the XGBoost Model!!!!")
        XGBoost_model = pickle.load(open(model_path_XG, 'rb'))
        models_list.append(XGBoost_model)
        logging.info("Finished Loading the XGBoost Model!!!!")
    except Exception as e:
        logging.error("Error in Loading the XGBoost Model!!!!")
        logging.error(e)
    
    # Score Directory

    score_dir = config["artifacts"]["report_dir"]
    scores_dir_path = os.path.join(artifacts_dir, score_dir)
    create_directory([scores_dir_path])

    scores_filename_rf = config["artifacts"]["rf_score"]
    scores_filename_xgb = config["artifacts"]["xgb_score"]

    scores_path_rf = os.path.join(artifacts_dir, score_dir, scores_filename_rf)
    scores_path_xgb = os.path.join(artifacts_dir, score_dir, scores_filename_xgb)



    for i in range(len(models_list)):
        model = models_list[i]
        predicted_values = model.predict(test_x)

        try:
            logging.info(f">>>>>>Started Evaluating the Model!!!!")
            accuracy, f1, roc_auc = evaluation_matrices(test_y, predicted_values)
            if i == 0:
                scores = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "roc_auc": roc_auc
                }
                save_reports(scores, scores_path_rf)
            else:
                scores = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "roc_auc": roc_auc
                }
                save_reports(scores, scores_path_xgb)
            logging.info(f"Finished Evaluating the {model} Model!!!!")
            logging.info("Accuracy: {}".format(accuracy))
            logging.info("F1 Score: {}".format(f1))
            logging.info("ROC AUC: {}".format(roc_auc))
        except Exception as e:
            logging.error(">>>>>Error in Evaluating the Model!!!!")
            logging.error(e)

    print("Evaluation Completed !!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file", default="config\config.yaml")
    parser.add_argument("--params", help="Path to the params file", default="params.yaml")
    args = parser.parse_args()

    try:
        logging.info(">>>>>>Started Stage 4 Evaluation !!!")
        evaluate(args.config, args.params)
        logging.info(">>>>>>Finished Stage 4 Evaluation !!!")
    except Exception as e:
        logging.error(">>>>>Error in Stage 4 Evaluation !!!")
        logging.error(e)
