from src.houseprice1.components.data_ingestion import DataIngestion
from src.houseprice1.components.data_transformation import DataTransformation
from src.houseprice1.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        ## Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print(train_data_path, test_data_path)

        ##Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path=train_data_path, test_path=test_data_path
        )

        ##Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)
