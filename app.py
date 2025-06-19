from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import joblib
import os
from tensorflow.keras import layers, models

app = Flask(__name__)

# ----------- Load Class Names -----------
class_names = [
    '1. Eczema 1677',
    '3. Atopic Dermatitis - 1.25k',
    '6. Benign Keratosis-like Lesions (BKL) 2624',
    '8. Seborrheic Keratoses and other Benign Tumors - 1.8k',
    '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k'
]

# ----------- CNN Feature Extractor -----------
def load_cnn_feature_extractor():
    paths = [
        r"D:\aiml_lab_el\cnn_feature_extractor_model2.h5",
        r"D:\aiml_lab_el\cnn_feature_extractor_model2.keras"
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                print(f"✅ Loaded CNN feature extractor from: {path}")
                return model
            except Exception as e:
                print(f"❌ Failed to load model from {path}: {e}")
    
    print("❌ No valid CNN model found.")
    return None


cnn_feature_extractor = load_cnn_feature_extractor()

# ----------- Ensemble Model -----------
def load_ensemble_model():
    model_path = r"D:\aiml_lab_el\ensemble_model.pkl"
    try:
        model = joblib.load(model_path)
        print("✅ Ensemble VotingClassifier loaded")
        return model
    except Exception as e:
        print(f"❌ Failed to load ensemble model: {e}")
        return None

ensemble_model = load_ensemble_model()

# ----------- Image Preprocessing -----------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# ----------- Feature Extraction -----------
def extract_features(image):
    features = cnn_feature_extractor.predict(image, verbose=0)
    return features.flatten().reshape(1, -1)

# ----------- Prediction -----------
def ensemble_predict(features):
    prediction = ensemble_model.predict(features)[0]

    if hasattr(ensemble_model, "predict_proba"):
        probabilities = ensemble_model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
    else:
        probabilities = np.zeros(len(class_names))
        probabilities[prediction] = 1.0
        confidence = 1.0

    return prediction, confidence

# ----------- API Routes -----------

@app.route("/")
def home():
    return send_from_directory('.', 'index.html')

@app.route("/styles.css")
def styles():
    return send_from_directory('.', 'styles.css')

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    try:
        image_bytes = file.read()
        processed_img = preprocess_image(image_bytes)

        # --- Extract features ---
        features = extract_features(processed_img)

        # --- Predict with ensemble ---
        prediction, confidence = ensemble_predict(features)
        predicted_label = class_names[prediction] if 0 <= prediction < len(class_names) else "Unknown"

        # --- Map to simplified disease names ---
        pred_map = {
            '1. Eczema 1677': 'Eczema',
            '3. Atopic Dermatitis - 1.25k': 'Atopic Dermatitis',
            '6. Benign Keratosis-like Lesions (BKL) 2624': 'Benign Keratosis',
            '8. Seborrheic Keratoses and other Benign Tumors - 1.8k': 'Seborrheic Keratoses',
            '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k': 'Tinea Ringworm Candidiasis'
        }

        simplified = pred_map.get(predicted_label, 'Unknown')

        response = {
            "class": simplified,
            "confidence": confidence
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "cnn_feature_extractor": "Loaded" if cnn_feature_extractor else "Not loaded",
        "ensemble_model": "Loaded" if ensemble_model else "Not loaded",
        "class_names": class_names
    })

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    print("✅ Application ready and running!")
    app.run(debug=True)
