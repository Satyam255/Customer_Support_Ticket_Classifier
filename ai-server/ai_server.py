import os
from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow Node.js server to communicate

# Path to your fine-tuned model
MODEL_PATH = "my-ticket-classifier-final"
classifier = None

print("--- Loading AI Model ---")

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model directory not found at '{MODEL_PATH}'")
else:
    try:
        classifier = pipeline("text-classification", model=MODEL_PATH, device=0)
        print("✅ Model loaded successfully on GPU (if available).")
    except Exception as e:
        print(f"⚠️ GPU load failed: {e} — falling back to CPU.")
        try:
            classifier = pipeline("text-classification", model=MODEL_PATH, device=-1)
            print("✅ Model loaded successfully on CPU.")
        except Exception as e:
            print(f"❌ FATAL: Could not load model: {e}")

@app.route("/classify", methods=["POST"])
def classify_text():
    if classifier is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text_to_classify = data["text"]
    try:
        result = classifier(text_to_classify)
        prediction = result[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return jsonify({"error": "Classification failed"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask API is working"}), 200

if __name__ == "__main__":
    print("🚀 Starting Flask AI server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

