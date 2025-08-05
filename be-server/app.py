import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from datetime import datetime
from pathlib import Path
from utils import training

from controller.captureImageController import capture_image
from utils.FuzzyLogic import CalculateSprayingDelay
from utils.adaptiveControll import AdaptiveControll

adap = AdaptiveControll()

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Custom environment variables
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))


@app.route("/")
def home():
    return f"Welcome to the Flask App!"


@app.route("/capture-image")
def capture_image_route():
    filepath = capture_image()
    if filepath:
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Success capturing image",
                }
            ),
            200,
        )
    else:
        return jsonify({"status": "failed", "message": "Failed to capture image"}), 500


@app.route("/delay-spray", methods=["POST"])
def delay_spray_route():
    air_temperature = request.json.get("temperature")
    humidity = request.json.get("humidity")
    if air_temperature is None or humidity is None:
        return (
            jsonify(
                {"status": "error", "message": "Missing airTemperature or humidity"}
            ),
            400,
        )

    delay = CalculateSprayingDelay(air_temperature, humidity)

    if delay is None:
        return jsonify({"status": "error", "message": "Failed to calculate delay"}), 500

    return jsonify({"status": "success", "delay": delay}), 200


@app.route("/train", methods=["POST"])
def train_model():
    try:
        start_training = datetime.now()

        training.training_model()

        training_duration = datetime.now() - start_training
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Training failed: {str(e)}",
                }
            ),
            500,
        )

    return (
        jsonify(
            {
                "status": "success",
                "message": "Training completed",
                "duration": str(training_duration),
            }
        ),
        200,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    start_prediction = datetime.now()

    try:
        prediction = adap.predict()

        if prediction is None:
            return (
                jsonify({"status": "error", "message": "Failed to get prediction"}),
                500,
            )

        # Save the prediction to a file
        filepath = Path("public/prediction/data.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists():
            prediction.to_csv(filepath, mode="a", header=False)
        else:
            prediction.to_csv(filepath)

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Prediction failed: {str(e)}",
                }
            ),
            500,
        )

    prediction_duration = datetime.now() - start_prediction

    return (
        jsonify(
            {
                "status": "success",
                "message": "Prediction completed",
                "duration": str(prediction_duration),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=app.config["DEBUG"])
