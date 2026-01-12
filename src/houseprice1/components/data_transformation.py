import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.houseprice1.utils import save_object, create_quality_weighted_area

from src.houseprice1.exception import CustomException
from src.houseprice1.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            ## Categorical columns
            ord_categorical_cols = ["ExterQual", "BsmtQual", "KitchenQual", "GarageFinish"]
            ranking_type_1 = ["Po", "Fa", "TA", "Gd", "Ex"]
            ranking_type_2 = ["Unf", "RFn", "Fin"]
            hierarchies = [ranking_type_1, ranking_type_1, ranking_type_1, ranking_type_2]

            one_categorical_cols = ["MSZoning", "CentralAir", "Neighborhood"]

            ## Numerical Columns
            numerical_cols = [
                "OverallQual",
                "GrLivArea",
                "TotalBsmtSF",
                "FullBath",
                "YearBuilt",
                "TotRmsAbvGrd",
                "Fireplaces",
                "WoodDeckSF",
                "OpenPorchSF",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("feature_engineering", create_quality_weighted_area()),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_ord_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(categories=hierarchies)),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            cat_onehot_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("one_hot_pipeline", cat_onehot_pipeline, one_categorical_cols),
                    ("ordinal_pipeline", cat_ord_pipeline, ord_categorical_cols),
                ]
            )

            return preprocessor

        except Exception as ex:
            raise CustomException(ex, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(os.path.join(train_path))
            test_df = pd.read_csv(os.path.join(test_path))

            logging.info("Reading data train and test file")

            preprocessing_obj = self.get_data_transformation_object()

            target_col_name = "SalePrice"

            input_features_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]
            target_feature_train_df = np.log1p(target_feature_train_df)

            input_features_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]
            target_feature_test_df = np.log1p(target_feature_test_df)

            logging.info("Applying Preprocessing on train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as ex:
            raise CustomException(ex, sys)
