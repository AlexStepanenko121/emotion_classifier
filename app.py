from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app) #Використовуємо Cross-Origin Resource Sharing

classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

@app.route('/api/classify', methods=['POST'])
def classify_text():
    try:
        data = request.json #Отримаємо запит з фронтенду
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400 #Помилка, якщо немає тексту

        predictions = classifier(text, top_k=None)
        emotion_probs = {pred['label']: pred['score'] for pred in predictions} #Викликаємо модель і рахуємо ймовірності

        total_score = sum(emotion_probs.values()) # Для простоти візьмемо середнє арифметичне емоцій
        normalized_probs = {k: v / total_score for k, v in emotion_probs.items()} #нормалізуємо, щоб сума ймовірностей була рівна 1

        return jsonify({'emotion_probabilities': normalized_probs}) #повертаємо ймовірності
    
    except Exception as e:
        print(f"Error: {str(e)}") # Якщо сталась помилка, виводимо її в термінал, а користувачу повертаємо, що щось пішло не так
        return jsonify({'error': 'Something went wrong'}), 500

if __name__ == '__main__':
    app.run(debug=True)