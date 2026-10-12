
import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class CustomData:
    """Store student information and convert it into a DataFrame."""

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """Create a DataFrame using the exact training column names."""

        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [
                    self.parental_level_of_education
                ],
                "lunch": [self.lunch],
                "test_preparation_course": [
                    self.test_preparation_course
                ],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys) from e


class PredictPipeline:
    """Load the trained model and predict the math score."""

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join(
                "artifacts", "preprocessor.pkl"
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )

            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found: {preprocessor_path}"
                )

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Check that the input columns match the training columns.
            print("Input columns:", features.columns.tolist())
            print(
                "Expected columns:",
                preprocessor.feature_names_in_.tolist()
            )

            transformed_data = preprocessor.transform(features)
            predictions = model.predict(transformed_data)

            return predictions

        except Exception as e:
            raise CustomException(e, sys) from e
