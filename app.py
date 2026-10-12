
from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

application = Flask(__name__)
app = application


@app.route("/")
def index():
    """Display the home page."""
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """Display the form and predict the student's math score."""

    if request.method == "GET":
        return render_template("home.html")

    try:
        # Read the submitted form values.
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=int(request.form.get("reading_score")),
            writing_score=int(request.form.get("writing_score"))
        )

        # Convert the input into a DataFrame.
        pred_df = data.get_data_as_data_frame()

        # Generate the prediction.
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display the predicted math score.
        return render_template(
            "home.html",
            results=round(float(results[0]), 2)
        )

    except Exception as e:
        # Display any prediction error on the form.
        return render_template(
            "home.html",
            error=str(e)
        ), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
