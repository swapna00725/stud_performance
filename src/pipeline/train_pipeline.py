import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=="__main__":
    try:
        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        modeltrainer=ModelTrainer()
        model_score=modeltrainer.initiate_model_trainer(train_arr,test_arr)

        logging.info("model training completed : {model_score}")

        print(f"final model score : {model_score}")
    except Exception as e :
        logging.error('training pipeline failed')
        raise CustomException(e,sys)