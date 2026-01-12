import os
import sys
import numpy as np
import pandas as pd
from src.houseprice1.exception import CustomException
from src.houseprice1.utils import load_object

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class PredictPipeline:
    def __init__(self):
        self.model_config = ModelConfig()

    def predict(self, features):
        try:
            model_path = self.model_config.model_path
            preprocessor_path = self.model_config.preprocessor_path

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_transformation = preprocessor.transform(features)
            preds_log = model.predict(data_transformation)

            final_prediction = np.expm1(preds_log)
            return final_prediction

        except Exception as ex:
            raise CustomException(ex, sys)
