from datasets import Dataset
import json
import torch

with open("annotated_slang_examples.json", "r") as f:
    annotated_data = json.load(f)


dataset = Dataset.from_list(annotated_data) # конвертуємо в датасет, який підтримує HuggingFace

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")


emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", 
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", 
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", 
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]
label2id = {emotion: idx for idx, emotion in enumerate(emotions)}
id2label = {idx: emotion for emotion, idx in label2id.items()}

def preprocess_function(examples):

    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    # Переводимо мітки з файла json в бінарний формат - тобто, 1, якщо емоція присутня, і 0 в протилежному випадку
    binary_labels = []
    for label_list in examples["labels"]: 
        label_indices = [label2id[label] for label in label_list]
        binary_vector = [1 if i in label_indices else 0 for i in range(len(emotions))]
        binary_labels.append(binary_vector)

    # додаємо бінарні мітки для вхідних данів і переводимо у тип float
    inputs["labels"] = [torch.tensor(vec, dtype=torch.float) for vec in binary_labels]
    return inputs



encoded_dataset = dataset.map(preprocess_function, batched=True)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Завантажуємо модель
model = AutoModelForSequenceClassification.from_pretrained(
    "SamLowe/roberta-base-go_emotions", 
    num_labels=len(emotions),
    id2label=id2label,
    label2id=label2id
)

# Параметри тренування
training_args = TrainingArguments(
    output_dir="./results", # зберігаємо результат у папку results
    evaluation_strategy="epoch", # робимо оцінку помилки моделі після кожної ітерації на всіх бетчах
    learning_rate=5e-5, # параметр альфа для градієнтного спуску
    per_device_train_batch_size=8, # розмір міні-бетча (розбиваємо датасет на масиви з 8 елементів і тренуємо на кожному з них)
    num_train_epochs=3, # кількість ітерацій по датасету для тренування
    weight_decay=0.01, # параметр для регулярізації, тобто, щоб запобігти overfitting
    save_total_limit=2, # зберігаємо 2 контрольні точки
    logging_dir='./logs', # зберігаємо параметри кожні 10 кроків 
    logging_steps=10,
    save_steps=500, # зберігаємо контрольні точки кожні 500 кроків 
    push_to_hub=False # не дозволяємо надсилати результат до Hugging Face
)
from sklearn.model_selection import train_test_split

# Розбиваємо датасет на тренування і тестування
train_texts, val_texts, train_labels, val_labels = train_test_split(
    [example["text"] for example in dataset], 
    [example["labels"] for example in dataset],
    test_size=0.1,
    random_state=42, # для рандомізації
)

# Переводимо словник в датасет
from datasets import Dataset
train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_val_dataset = val_dataset.map(preprocess_function, batched=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_val_dataset,
    tokenizer=tokenizer
)

# Тренуємо модель
trainer.train()

model.save_pretrained("./fine_tuned_emotion_model")
tokenizer.save_pretrained("./fine_tuned_emotion_model")
