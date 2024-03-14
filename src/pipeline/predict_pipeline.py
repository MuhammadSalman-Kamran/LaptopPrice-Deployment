from src.exception import CustomException
from src.logger import logging
import sys
import os
import pandas as pd
from src.utils import load_obj


class Prediction:
    def __init__(self) -> None:
        pass

    def prediction(self, input):
        model_file_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')
        model = load_obj(model_file_path)
        preprocessor = load_obj(preprocessor_file_path)
        processed_input = preprocessor.transform(input)
        pred = model.predict(processed_input)
        return pred



class CustomDataClass:
    def __init__(self, company, type_name, ram, weight, touch_screen, ips, ppi,cpu_brand, hdd, ssd, gpu_brand, os):
        self.company = company
        self.type_name = type_name
        self.ram = ram
        self.weight = weight
        self.touch_screen = touch_screen
        self.ips = ips
        self.ppi = ppi
        self.cpu_brand = cpu_brand
        self.hdd = hdd
        self.ssd = ssd
        self.gpu_brand = gpu_brand
        self.os = os

    def data_as_df(self):
        return (pd.DataFrame([[self.company, self.type_name, self.ram, self.weight, self.touch_screen, self.ips, self.ppi,self.cpu_brand, self.hdd, self.ssd, self.gpu_brand, self.os]], columns=['Company','TypeName','Ram',	'Weight', 'TouchScreen','Ips','ppi','Cpu Brand','HDD','SSD','Gpu Brand','OS']))