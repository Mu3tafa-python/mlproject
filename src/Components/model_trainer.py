
import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        preprocessor_path=None
    ):
        try:
            logging.info("Splitting training and testing data")

            # Separate input features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define regression models
            models = {
                "Linear Regression": LinearRegression(),

                "K-Neighbors Regressor": KNeighborsRegressor(),

                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),

                "Random Forest": RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                ),

                "AdaBoost": AdaBoostRegressor(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42
                ),

                "XGBoost RF Regressor": XGBRFRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
            }

            # Train and evaluate the models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            if not model_report:
                raise ValueError(
                    "No model evaluation results were produced"
                )

            # Select the model with the highest test R2 score
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model R2: {best_model_score}")

            print("\nBest Model:", best_model_name)
            print("Best Model R2 Score:", round(best_model_score, 4))

            # Check the minimum acceptable score
            if best_model_score < 0.6:
                raise ValueError(
                    "No model reached the minimum R2 score of 0.6"
                )

            # Save the selected trained model
            save_object(
                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                obj=best_model
            )

            logging.info("Best model saved successfully")

            # Calculate the final test R2 score
            predicted = best_model.predict(X_test)
            test_r2 = r2_score(y_test, predicted)

            return test_r2

        except Exception as e:
            logging.exception("Error during model training")
            raise CustomException(e, sys) from e
