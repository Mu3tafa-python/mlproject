
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts", "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts", "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts", "data.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            # Load the original dataset
            df = pd.read_csv("notebook/data/stud.csv")

            logging.info("Dataset loaded successfully")

            # Create the artifacts directory
            os.makedirs("artifacts", exist_ok=True)

            # Save the original dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save the training dataset
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            # Save the testing dataset
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.exception("Error during data ingestion")
            raise CustomException(e, sys) from e


if __name__ == "__main__":

    try:
        # Step 1: Data Ingestion
        print("\n========== STEP 1: DATA INGESTION ==========")

        ingestion = DataIngestion()

        train_data, test_data = (
            ingestion.initiate_data_ingestion()
        )

        print("Data ingestion completed successfully!")
        print("Training data:", train_data)
        print("Testing data:", test_data)

        # Step 2: Data Transformation
        print("\n========== STEP 2: DATA TRANSFORMATION ==========")

        transformation = DataTransformation()

        train_arr, test_arr, preprocessor_path = (
            transformation.initiate_data_transformation(
                train_data,
                test_data
            )
        )

        print("Data transformation completed successfully!")
        print("Training array shape:", train_arr.shape)
        print("Testing array shape:", test_arr.shape)
        print("Preprocessor saved at:", preprocessor_path)

        # Step 3: Model Training
        print("\n========== STEP 3: MODEL TRAINING ==========")
        print("DEBUG: Starting model training...")

        model_trainer = ModelTrainer()

        print("DEBUG: ModelTrainer object created.")
        print("DEBUG: Calling initiate_model_trainer...")

        test_r2 = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr,
            preprocessor_path
        )

        print("\nModel training completed successfully!")
        print("Final Test R2 Score:", round(test_r2, 4))

        # Verify that the model file exists
        model_path = os.path.join(
            "artifacts", "model.pkl"
        )

        if os.path.exists(model_path):
            print("Trained model saved at:", model_path)
            print(
                "Model file size:",
                os.path.getsize(model_path),
                "bytes"
            )
        else:
            raise FileNotFoundError(
                f"Model file was not created: {model_path}"
            )

    except Exception as e:
        logging.exception("Pipeline execution failed")
        print("\nPIPELINE FAILED!")
        print(f"Error details: {e}")
        raise
