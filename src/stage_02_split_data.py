from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'split_data.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def split_and_save(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # Read data
    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']
    raw_local_file = config['artifacts']['raw_local_file']

    raw_local_file_path = os.path.join(artifacts_dir, raw_local_dir, raw_local_file)

    df = pd.read_csv(raw_local_file_path)

    # Split data

    split_ratio = params['base']['test_size']
    random_state = params['base']['random_state']

    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)

    split_data_dir = config['artifacts']['split_data_dir']

    create_directory([os.path.join(artifacts_dir, split_data_dir)])
    
    train_data_filename = config['artifacts']['train']
    test_data_filename = config['artifacts']['test']

    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)

    for data, data_path in (train, train_data_path), (test, test_data_path):
        save_local_df(data, data_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', "-c", type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--params', "-p", type= str, default='params.yaml', help='Path to params file')
    parsed_args = parser.parse_args()


    try:
        logging.info(">>>>>>Stage 2 started Split data")
        split_and_save(parsed_args.config, parsed_args.params)
        logging.info(">>>>>>Stage 2 completed Split data and saved to file path!!!!!")

    except Exception as e:
        logging.error(">>>>>>Stage 2 failed Split data")
        logging.error(e)



