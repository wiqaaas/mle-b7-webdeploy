# flask_app.py
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({"message": "Welcome to Flask"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Simulate a prediction by sleeping for 2 seconds.
    """
    data = request.get_json()
    number = data["number"]
    time.sleep(2)  # Simulate a long-running process
    result = {"input": number, "prediction": number * 2}
    return jsonify(result)

if __name__=="__main__":
	app.run(host="0.0.0.0", port=5000)
