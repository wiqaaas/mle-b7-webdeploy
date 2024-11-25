from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import io

app = Flask(__name__)

# Load pretrained models
tabular_model = joblib.load("models/tabular_model.pkl")
text_model = tf.keras.models.load_model("models/text_model.h5")
image_model = tf.keras.models.load_model("models/image_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict/tabular", methods=["POST"])
def predict_tabular():
    """
    API to predict using tabular data.
    Example cURL:
    curl -X POST -H "Content-Type: application/json" \
        -d '[{"feature1": 1}, {"feature2": 2}]' \
        http://127.0.0.1:5000/predict/tabular
    """
    data = request.json
    df = pd.DataFrame(data)
    predictions = tabular_model.predict(df)
    return jsonify({"predictions": predictions.tolist()})

@app.route("/predict/text", methods=["POST"])
def predict_text():
    """
    API to predict using text data.
    Example cURL:
    curl -X POST -H "Content-Type: application/json" \
        -d '{"text": "sample text"}' \
        http://127.0.0.1:5000/predict/text
    """
    text = request.json.get("text", "")
    dummy_input = np.random.randint(1000, size=(1, 10))  # Random input for dummy model
    predictions = text_model.predict(dummy_input)
    return jsonify({"predictions": predictions.tolist()})

@app.route("/predict/image", methods=["POST"])
def predict_image():
    """
    API to predict using image data.
    Example cURL:
    curl -X POST -F "image=@sample.jpg" http://127.0.0.1:5000/predict/image
    """
    image_file = request.files.get("image")
    image = Image.open(io.BytesIO(image_file.read())).resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    predictions = image_model.predict(image_array)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
