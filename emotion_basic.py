import time
from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("SetFit/emotion", split="test")

classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

start_time = time.time()

correct = 0
total = len(dataset)
print ("Total length of the dataset: ",total)
for example in dataset:
    text = example["text"]
    true_label = example["label"]

    prediction = classifier(text)
    predicted_label = emotion_labels.index(prediction[0][0]['label']) 
    """модель видає результат у вигляді [[{"label": label, 'score': score}]], 
    причому перший label відповідає домінуючій емоції, тому виділяємо його"""

    if predicted_label == true_label:
        correct += 1

end_time = time.time()

accuracy = correct / total

print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Accuracy: {accuracy * 100:.2f}%")
