import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("========== Training Pipeline Started ==========")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_data, test_data = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_data, test_data)

        # Step 3: Model Training
        trainer = ModelTrainer()
        model_score = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"========== Training Pipeline Completed | Model Score: {model_score} ==========")
        print(f"✅ Model Training Completed | Score: {model_score}")

    except Exception as e:
        logging.error("Pipeline failed due to an error")
        raise CustomException(e, sys)


  