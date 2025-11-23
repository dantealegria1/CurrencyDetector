from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# --- RUTAS DEL MODELO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "training", "usd_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "..", "training", "labels.txt")

# --- Cargar modelo y etiquetas ---
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = [l.strip() for l in f.readlines()]

IMAGE_SIZE = 224

def classify_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    idx = np.argmax(prediction)
    confidence = float(prediction[0][idx])

    return labels[idx], confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    upload_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(upload_dir, exist_ok=True)  # Por si no existe

    upload_path = os.path.join(upload_dir, "upload.jpg")
    file.save(upload_path)

    label, confidence = classify_image(upload_path)

    return jsonify({
        "class": label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
