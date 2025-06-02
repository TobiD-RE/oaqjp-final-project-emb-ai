"""
Flask app to detect emotions from given text using EmotionDetection module.
"""

from flask import Flask, request, jsonify
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route('/emotionDetector', methods=['POST'])
def detect_emotion():
    """
    Endpoint to analyze emotion in given text.

    Expects JSON with key 'text'. Returns emotion scores or error messages.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request'}), 400

    result = emotion_detector(data['text'])

    if result.get('dominant_emotion') is None:
        return jsonify({'message': 'Invalid text! Please try again!'}), 400

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)
