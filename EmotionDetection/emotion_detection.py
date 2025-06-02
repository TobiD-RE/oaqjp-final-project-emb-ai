from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

def emotion_detector(text_to_analyse):

    if not text_to_analyse or text_to_analyse.strip() == "":
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    input_json = { "raw_document": { "text": text_to_analyse } }

    try:
        response = requests.post(url, headers=headers, json=input_json, timeout=10)
    except requests.exceptions.RequestException:
        # On any request error, return dict with None values
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    if response.status_code == 400:
        # If bad request, return None dict
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
    elif response.status_code != 200:
        # For other error statuses, also return None dict
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    response_dict = response.json()

    emotions = response_dict['emotionPredictions'][0]['emotion']

    anger_score = emotions.get('anger')
    disgust_score = emotions.get('disgust')
    fear_score = emotions.get('fear')
    joy_score = emotions.get('joy')
    sadness_score = emotions.get('sadness')

    all_emotions = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }
    dominant_emotion = max(all_emotions, key=all_emotions.get) if all_emotions else None

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }


@app.route('/emotionDetector', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request'}), 400
    
    result = emotion_detector(data['text'])

    # If all emotion values are None, treat as bad input or error
    if all(value is None for key, value in result.items() if key != 'dominant_emotion'):
        return jsonify({'error': 'Invalid input or external API error'}), 400
    
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)