
import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """Save a Python object to a file using pickle."""
    try:
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params
):
    """Tune, train, and evaluate multiple regression models."""
    try:
        report = {}

        for model_name, model in models.items():
            print(f"\n{'=' * 60}")
            print(f"Model: {model_name}")
            print("Starting hyperparameter tuning...")

            # Get the parameter grid for the current model
            model_params = params.get(model_name, {})

            if model_params:
                # Perform grid search with three-fold cross-validation
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    scoring="r2",
                    n_jobs=1,
                    refit=True,
                    verbose=0
                )

                # Train and find the best parameter combination
                grid_search.fit(X_train, y_train)

                # Select the best estimator
                best_model = grid_search.best_estimator_

                print("Best Parameters:", grid_search.best_params_)
                print(
                    "Best Cross-Validation R2:",
                    round(grid_search.best_score_, 4)
                )

            else:
                # Train the model directly if no grid is provided
                best_model = model
                best_model.fit(X_train, y_train)

            # Evaluate the selected model on the test set
            y_test_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)

            print("Test R2 Score:", round(test_r2, 4))

            # Keep the tuned model for later selection
            models[model_name] = best_model
            report[model_name] = test_r2

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
