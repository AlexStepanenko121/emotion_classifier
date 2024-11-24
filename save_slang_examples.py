import os
import json
from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("MLBtrio/genz-slang-dataset")

classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

output_file = "annotated_slang_examples.json"

# Завантажуємо і зчитуємо дані з файла json, які там вже є, щоб потім не повторюватися
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        try:
            annotated_data = json.load(f) or []
        except json.JSONDecodeError:
            print(f"Warning: {output_file} is corrupted or empty. Starting fresh.")
            annotated_data = []
    processed_texts = {item["text"] for item in annotated_data}
    print(f"Loaded {len(annotated_data)} previously annotated examples.")
else:
    annotated_data = []
    processed_texts = set()

try:
    for item in dataset["train"]:
        description = item["Description"]  # Див. датасет (8) - це стовпчик з описами кожного сленга
        example_sentence = item["Example"] # Приклад з уживанням цього сленга
        
        if not description or not example_sentence:
            continue

        if example_sentence in processed_texts:
            continue

        description_predictions = classifier(description, top_k=None)
        description_emotion = max(description_predictions, key=lambda x: x["score"])["label"]

        print(f"\nDescription: {description}")
        print(f"Predicted Emotion for Description: {description_emotion}")

        example_predictions = classifier(example_sentence, top_k=None)
        example_emotion = max(example_predictions, key=lambda x: x["score"])["label"]

        print(f"Example Sentence: {example_sentence}")
        print(f"Predicted Emotion for Example: {example_emotion}")

        # Якщо я згодний з тим, що видає емоція, я натискаю Enter, а якщо ні, я вводжу правильний варіант, який потім зберігається у json
        correct_emotions = input("Correct emotions (comma-separated, or press Enter to accept model's prediction): ").strip()
        if correct_emotions:
            labels = [label.strip() for label in correct_emotions.split(",")]
        else:
            labels = [example_emotion]

        annotated_data.append({"text": example_sentence, "labels": labels})
        processed_texts.add(example_sentence)

        # Зберігаємо у файл json
        with open(output_file, "w") as f:
            json.dump(annotated_data, f)

        print(f"Saved {len(annotated_data)} examples.")

except KeyboardInterrupt:
    print("\nAnnotation interrupted. Progress saved.")

