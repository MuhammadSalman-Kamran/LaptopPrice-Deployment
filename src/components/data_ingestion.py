import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import train_test_split_function
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

@dataclass
class DataIngestionConfig:
    sample_data_file_path = os.path.join('artifacts','sample.csv')
    train_data_file_path = os.path.join('artifacts', 'train.csv')
    test_data_file_path = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()

    def ingest_data(self):
        logging.info('Welcome to the initial Data Ingestion Phase of the data!')
        try:
            logging.info('Reading the data into DataFrame')
            df = pd.read_csv('notebook/data/Clean_Laptop.csv')

            logging.info('Making the artifacts Directory')
            os.makedirs(os.path.dirname(self.data_ingestion_config.sample_data_file_path), exist_ok=True)

            logging.info('Exporting the sample csv file into the artifacts folder')
            df.to_csv(self.data_ingestion_config.sample_data_file_path, index=False, header = True)

            logging.info('Calling the Train Test Split function from utils')
            train_df, test_df = train_test_split_function(df, 0.2)

            train_df.to_csv(self.data_ingestion_config.train_data_file_path, index = False, header = True)
            logging.info('Training file has exported to artifacts folder')

            test_df.to_csv(self.data_ingestion_config.test_data_file_path, index = False, header = True)
            logging.info('Testing file has exported to artifacts folder')

            return (
                self.data_ingestion_config.train_data_file_path,
                self.data_ingestion_config.test_data_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_df, test_df = obj.ingest_data()

    transformation_obj = DataTransformation()
    train_input_arr, test_input_arr = transformation_obj.init_transform(train_df, test_df)

    train_model_obj = ModelTraining()
    train_model_obj.init_training(train_df, train_input_arr,test_df, test_input_arr)