"""
Fake News Detector - Flask Backend
"""

import os
import pickle
import logging
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
pipeline = None


def load_model():
    """Load the trained ML pipeline from disk."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "model.pkl not found. Training a new model from synthetic data..."
        )
        # Auto-train on first run if no model exists
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
            from train_model import train_and_save
            pipeline = train_and_save()
            logger.info("Model trained and saved successfully.")
        except Exception as e:
            logger.error(f"Failed to auto-train model: {e}")
            pipeline = None
        return

    try:
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        logger.info("Model loaded successfully from model.pkl")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        pipeline = None


load_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Basic text cleaning before prediction."""
    return text.strip()


def get_confidence_label(probability: float) -> str:
    if probability >= 0.90:
        return "Very High"
    elif probability >= 0.75:
        return "High"
    elif probability >= 0.60:
        return "Moderate"
    else:
        return "Low"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON): { "text": "<news article text>" }
    Returns:
    {
        "prediction": "REAL" | "FAKE",
        "confidence": 0.92,
        "confidence_label": "Very High",
        "real_probability": 0.92,
        "fake_probability": 0.08
    }
    """
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Please try again later."}), 503

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = preprocess(data["text"])
    if len(text) < 10:
        return jsonify({"error": "Text is too short. Please enter a proper news article."}), 400
    if len(text) > 50000:
        return jsonify({"error": "Text is too long (max 50,000 characters)."}), 400

    try:
        prediction = pipeline.predict([text])[0]

        # Get probability scores if the classifier supports it
        try:
            proba = pipeline.predict_proba([text])[0]
            classes = pipeline.classes_
            proba_dict = dict(zip(classes, proba))
            real_prob = float(proba_dict.get("REAL", 0.5))
            fake_prob = float(proba_dict.get("FAKE", 0.5))
        except AttributeError:
            # PassiveAggressiveClassifier uses decision_function instead
            decision = pipeline.decision_function([text])[0]
            # Convert decision score to a probability-like value using sigmoid
            import math
            sigmoid = lambda x: 1 / (1 + math.exp(-x))
            classes = list(pipeline.classes_)
            if "REAL" in classes:
                # Positive decision = REAL for PAC
                real_prob = sigmoid(decision)
                fake_prob = 1 - real_prob
            else:
                fake_prob = sigmoid(decision)
                real_prob = 1 - fake_prob

        confidence = real_prob if prediction == "REAL" else fake_prob
        confidence_label = get_confidence_label(confidence)

        logger.info(
            f"Prediction: {prediction} | Confidence: {confidence:.2%} | "
            f"Text snippet: {text[:60]}..."
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "confidence_label": confidence_label,
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render.com."""
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
    }), 200


@app.route("/api/info", methods=["GET"])
def api_info():
    """Returns API metadata."""
    return jsonify({
        "name": "Fake News Detector API",
        "version": "1.0.0",
        "description": "Detects fake news using a TF-IDF + Machine Learning pipeline.",
        "endpoints": {
            "POST /predict": "Predict if news is REAL or FAKE",
            "GET /health": "Health check",
            "GET /api/info": "API information",
        },
        "model_loaded": pipeline is not None,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
