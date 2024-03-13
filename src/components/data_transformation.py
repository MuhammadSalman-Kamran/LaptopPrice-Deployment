import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def make_pipeline(self):
        logging.info('Making the pipeline for preprocessing')

        try:
            logging.info('Defining the numerical and categorical columns ')
            numerical_col = ['Ram','Weight','TouchScreen','Ips','ppi', 'HDD', 'SSD']
            categorical_col = ['Company','TypeName', 'Cpu Brand','Gpu Brand', 'OS']

            logging.info('Making the Pipeline for numerical values')
            numerical_pipe = Pipeline([
                ('Imputing', SimpleImputer(strategy='mean')),
                ('Scaling', StandardScaler())
            ])


            logging.info('Making the pipeline for categorical values')
            categorical_pipe = Pipeline([
                ('imputing', SimpleImputer(strategy='most_frequent')),
                ('encoding', OneHotEncoder())
            ])

            logging.info('Making the preprocessor, which will handle both numerical and categorical pipelines')
            preprocessor = ColumnTransformer([
                ('numerica_pipeline', numerical_pipe, numerical_col),
                ('categorical_pipe', categorical_pipe, categorical_col)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def init_transform(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Training and Testing files have imported')

            preprocessor_obj = self.make_pipeline()
            logging.info('Preprocessor object also imported')

            logging.info('Splitting the input and output Columns')
            train_input_before_process = train_df.drop(['Price'], axis = 1)
            # train_target = train_df['Price']
            test_input_before_process = test_df.drop(['Price'], axis = 1)
            # test_target = test_df['Price']

            logging.info('Preprocessing the input features')
            processed_train_input = preprocessor_obj.fit_transform(train_input_before_process)
            processed_test_input = preprocessor_obj.transform(test_input_before_process)
            logging.info('Preprocessing has done successfully!')
            
            # logging.info('Now Concatenating the processed input and output columns')
            # train_target = np.array(train_target)
            # train_target = train_target.reshape(-1,1)
            # print(train_input_after_process)
            # train_arr = np.concatenate((train_input_after_process, train_target), axis=1)
            # test_arr = np.c_[test_input_after_process, np.array(test_target)]

            logging.info('Calling the function to save the obj for preprocessing')
            save_obj(self.data_transformation_config.preprocessor_file_path, preprocessor_obj)

            return (processed_train_input, processed_test_input)

        except Exception as e:
            raise CustomException(e, sys)