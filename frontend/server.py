import os

from flask import Flask, Response, jsonify, request, send_from_directory

from onnx_inference import InferenceClient
from onnx_inference._exceptions import InferenceError

GATEWAY = os.getenv("GATEWAY_ADDRESS", "localhost:50050")

app = Flask(__name__)


@app.route("/")
def index() -> Response:
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def generate() -> Response:
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    max_new_tokens = int(data.get("max_new_tokens") or 0)
    temperature = float(data.get("temperature") or 0.0)
    try:
        with InferenceClient(GATEWAY) as client:
            text = client.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        return jsonify({"text": text})
    except InferenceError as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/synthesize", methods=["POST"])
def synthesize() -> Response:
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    speed = float(data.get("speed") or 1.0)
    try:
        with InferenceClient(GATEWAY) as client:
            audio = client.synthesize(text, speed=speed)
        return Response(audio, mimetype="audio/wav")
    except InferenceError as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/embed", methods=["POST"])
def embed() -> Response:
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        with InferenceClient(GATEWAY) as client:
            vector = client.embed(text)
        return jsonify({"vector": vector, "dimensions": len(vector)})
    except InferenceError as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/ocr", methods=["POST"])
def ocr() -> Response:
    if "image" not in request.files:
        return jsonify({"error": "image file required"}), 400
    image_bytes = request.files["image"].read()
    try:
        with InferenceClient(GATEWAY) as client:
            results = client.recognize(image_bytes)
        return jsonify({
            "results": [
                {"text": r.text, "confidence": r.confidence, "box": list(r.box)}
                for r in results
            ]
        })
    except InferenceError as exc:
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
