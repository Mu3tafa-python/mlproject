
import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define numerical and categorical columns
            numerical_columns = [
                "reading_score",
                "writing_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    ),
                    (
                        "scaler",
                        StandardScaler(with_mean=False)
                    )
                ]
            )

            logging.info("Numerical pipeline created")
            logging.info("Categorical pipeline created")

            # Combine the pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Training and testing data loaded")

            # Define the target column
            target_column_name = "math_score"

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )
            target_feature_test_df = test_df[target_column_name]

            # Create the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Fit on training data and transform it
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Transform testing data using the fitted object
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Convert sparse matrices to NumPy arrays if necessary
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()

            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Append target values to transformed features
            train_arr = np.c_[
                input_feature_train_arr,
                np.asarray(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.asarray(target_feature_test_df)
            ]

            # Save the fitted preprocessing object
            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config
                .preprocessor_obj_file_path
            )

        except Exception as e:
            logging.exception("Error during data transformation")
            raise CustomException(e, sys) from e
