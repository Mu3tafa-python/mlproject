
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
from xgboost import XGBRegressor
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
            logging.info(
                "Starting model training and hyperparameter tuning"
            )

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
                    n_jobs=1
                ),

                "AdaBoost": AdaBoostRegressor(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42,
                    thread_count=1
                ),

                "XGBoost Regressor": XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0
                )
            }

            # Define hyperparameter grids
            params = {
                "Linear Regression": {
                    "fit_intercept": [True, False]
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                },

                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "absolute_error"
                    ],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                },

                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5]
                },

                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.5, 1.0]
                },

                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 3]
                },

                "CatBoost Regressor": {
                    "iterations": [100, 200],
                    "depth": [4, 6],
                    "learning_rate": [0.03, 0.1]
                },

                "XGBoost Regressor": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1]
                }
            }

            # Tune and evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            if not model_report:
                raise ValueError(
                    "No model evaluation results were produced."
                )

            # Select the model with the highest test R2 score
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")
            logging.info(
                f"Best test R2 score: {best_model_score}"
            )

            print("\n========== BEST MODEL ==========")
            print("Best Model:", best_model_name)
            print(
                "Best Test R2 Score:",
                round(best_model_score, 4)
            )

            # Check the minimum acceptable score
            if best_model_score < 0.6:
                raise ValueError(
                    "No model reached the minimum R2 score of 0.6."
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

            print(
                "Final Test R2 Score:",
                round(test_r2, 4)
            )
            print(
                "Saved model:",
                self.model_trainer_config.trained_model_file_path
            )

            return test_r2

        except Exception as e:
            logging.exception("Error during model training")
            raise CustomException(e, sys) from e
