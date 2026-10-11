
import os
import sys
import pickle

from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        # Create the parent directory if it does not exist
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save the object
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models
):
    """
    Train each regression model and evaluate it using R2 score.
    """
    try:
        report = {}

        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Predict test results
            y_test_pred = model.predict(X_test)

            # Calculate R2 score
            score = r2_score(y_test, y_test_pred)

            # Store the score
            report[model_name] = score

            print(f"{model_name}: R2 Score = {score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
