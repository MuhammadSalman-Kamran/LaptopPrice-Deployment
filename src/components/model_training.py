import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from sklearn.ensemble import RandomForestRegressor
from src.utils import evaluate_model, save_obj

@dataclass
class ModelTrainConfig:
    model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTraining:
    def __init__(self) -> None:
        self.model_train_config = ModelTrainConfig()
    
    def init_training(self,train_path, train_input_arr,test_path, test_input_arr):
        try:
            df_train = pd.read_csv(train_path)
            train_target = df_train['Price']
            df_test = pd.read_csv(test_path)
            test_target = df_test['Price']
            model = RandomForestRegressor(n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)

            r_score = evaluate_model(model,train_input_arr, train_target, test_input_arr, test_target)
            print(r_score)

            save_obj(self.model_train_config.model_file_path ,model)

            return self.model_train_config.model_file_path
        except Exception as e:
            raise CustomException(e, sys)

