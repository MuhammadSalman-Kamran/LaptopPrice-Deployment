from sklearn.model_selection import train_test_split
import pickle
import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_absolute_error, r2_score

def train_test_split_function(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return (train_df, test_df)

def save_obj(file_path, obj):
    try:
        logging.info('Making the directory for save the object')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logging.info('Saving the object into the directory')
        pickle.dump(obj, open(file_path, 'wb'))

    except Exception as e:
        raise CustomException(e, sys)
    
def load_obj(file_path):
    try:
        logging.info('Starting loading the object')
        return pickle.load(open(file_path, 'rb'))
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(model, train_input, train_output, test_input, test_output):
    try:
        logging.info('Model Training is going to start')
        model.fit(train_input, train_output)
        prediction = model.predict(test_input)
        logging.info('Checking the error rate of the model')
        # error_score = mean_absolute_error(test_output, prediction)
        r_score = r2_score(test_output, prediction)

        return r_score
    
    except Exception as e:
        raise CustomException(e, sys)
