from numpy.lib.shape_base import split
from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def training_data(config_path, params_path):
    """
    This function is used to train the model.
    :param config_path: path to the config file
    :param params_path: path to the params file
    :return:
    """
    # Read the config file
    config = read_yaml(config_path)
    # Read the params file
    params = read_yaml(params_path)

    # Read the data
    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']
    train_file_name = config['artifacts']['train']

    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_file_name)

    # Read the data
    train_data = pd.read_csv(train_data_path)

    # Train Data

    train_y = train_data['Loan_Status']
    train_X = train_data.drop(['Loan_Status'], axis=1)

    # Load Parameters for Random Forest

    n_estimators = params['model_params']['random_forest']['n_estimators']
    max_depth = params['model_params']['random_forest']['max_depth']
    max_features = params['model_params']['random_forest']['max_features']
    min_samples_split = params['model_params']['random_forest']['min_samples_split']
    min_samples_leaf = params['model_params']['random_forest']['min_samples_leaf']
    model_name = params['model_params']['random_forest']['model_name']

    random_state= params['base']['random_state']

    # Create the model
    model = RandomForestClassifier(n_estimators= n_estimators,
                                   max_depth= max_depth,
                                   max_features= max_features,
                                   min_samples_leaf= min_samples_leaf,
                                   min_samples_split= min_samples_split,
                                   random_state= random_state)
    # Train the model
    model.fit(train_X, train_y)

    # Save the model
    model_dir = config["artifacts"]["model_dir"]
    model_filename = config["artifacts"]["model_file_random_forest"]

    model_dir = os.path.join(artifacts_dir, model_dir)

    create_directory([model_dir])

    model_path = os.path.join(model_dir, model_filename)

    pickle.dump(model, open(model_path, 'wb'))

    logging.info(f"Training completed for {model_name} !!!!!")


    # Load Parameters for Xgboost
    model_name = params['model_params']['xgboost']['model_name']
    n_estimators = params['model_params']['xgboost']['n_estimators']
    max_depth = params['model_params']['xgboost']['max_depth']
    learning_rate = params['model_params']['xgboost']['learning_rate']
    min_child_weight = params['model_params']['xgboost']['min_child_weight']
    gamma = params['model_params']['xgboost']['gamma']
    subsample = params['model_params']['xgboost']['subsample']
    colsample_bytree = params['model_params']['xgboost']['colsample_bytree']
    max_delta_step = params['model_params']['xgboost']['max_delta_step']

    # Create the model
    model = XGBClassifier(n_estimators= n_estimators,
                          max_depth= max_depth,
                          learning_rate= learning_rate,
                          min_child_weight= min_child_weight,
                          gamma= gamma,
                          subsample= subsample,
                          colsample_bytree= colsample_bytree,
                          max_delta_step= max_delta_step)
    
    model.fit(train_X, train_y)

    # Save the model
    model_dir = config["artifacts"]["model_dir"]
    model_filename = config["artifacts"]["model_file_xgboost"]

    model_dir = os.path.join(artifacts_dir, model_dir)

    create_directory([model_dir])

    model_path = os.path.join(model_dir, model_filename)

    pickle.dump(model, open(model_path, 'wb'))

    logging.info(f"Training completed for {model_name} !!!!!")
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',type=str, default='config/config.yaml')
    parser.add_argument('--params', '-p', type=str, default='params.yaml')
    args = parser.parse_args()

    try:
        logging.info(">>>>>>> Stage 3 Started")
        training_data(args.config, args.params)
        logging.info(">>>>>>> Stage 3 Completed, Training Completed !!!!!")
    except Exception as e:
        logging.error(">>>>>>> Stage 3 Failed")
        logging.error(e)
