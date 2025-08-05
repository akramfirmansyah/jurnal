import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from controller.captureImageController import capture_image
from controller.delayingSprayController import delaySprayController

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

    response = delaySprayController(airTemperature=air_temperature, humidity=humidity)
    if response is None:
        return jsonify({"status": "error", "message": "Failed to calculate delay"}), 500

    return jsonify({"status": "success", "delay": response}), 200


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=app.config["DEBUG"])
