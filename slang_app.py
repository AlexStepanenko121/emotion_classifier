from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

classifier = pipeline("text-classification", model="./fine_tuned_emotion_model") # це єдина зміна в порівнянні з app.py

@app.route('/api/classify', methods=['POST'])
def classify_text():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400 

        predictions = classifier(text, top_k=None)
        emotion_probs = {pred['label']: pred['score'] for pred in predictions} 

        total_score = sum(emotion_probs.values())
        normalized_probs = {k: v / total_score for k, v in emotion_probs.items()}

        return jsonify({'emotion_probabilities': normalized_probs})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Something went wrong'}), 500

if __name__ == '__main__':
    app.run(debug=True)
