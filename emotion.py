import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

start_time = time.time()


dataset_path = "hf://datasets/google-research-datasets/go_emotions/raw/train-00000-of-00001.parquet" # Датасет


data = pd.read_parquet(dataset_path) # зчитуємо датасет за допомогою pandas


import nltk
from nltk.corpus import stopwords
from textblob import Word
import re

data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split())) # Всі слова з маленької літери

data['text'] = data['text'].str.replace('[^\w\s]', ' ', regex=True) # Прибираємо пунктуацію

stop = stopwords.words('english') # Прибираємо stopwords (такі, як, наприклад, a, the, is, and...)
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

data['text'] = data['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) # лемматизація - всі дієслова ставимо в початкову форму

def de_repeat(text): # якщо раптом слово було написано з зайвим повтором літери, то зайві літери прибираємо
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)
data['text'] = data['text'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))


freq = pd.Series(' '.join(data['text']).split()).value_counts()[-10000:] # Приберемо слова, які є одними з 10000 найменш уживаних в англійській мові 
freq = list(freq.index)
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))


emotion_list = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Обробляємо датасет, бо якщо подивитись на нього, то для кожного прикладу (training example) на виході є бінарний вектор, нам його потрібно перетворити на одну мітку
emotion_columns = [col for col in data.columns if col in emotion_list]
data['emotion'] = data[emotion_columns].idxmax(axis=1)

from sklearn import preprocessing
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data['emotion'].values)

# беремо 70% датасету для тренування і 30% для тестування
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data.text.values, y, stratify=y, random_state=None, test_size=0.3, shuffle=True)

tfidf_vect = TfidfVectorizer(analyzer='word')

# Використовуємо метод TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vect.fit(data['text'])


X_train_tfidf = tfidf_vect.transform(X_train)
X_val_tfidf = tfidf_vect.transform(X_val)
# Використовуємо наївний Баєсів класифікатор
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_val_tfidf)


print('Naive Bayes TF-IDF Accuracy: %s' % accuracy_score(y_pred, y_val)) # рахуємо точність

# Рахуємо метрики precision, recall, f1-score для більш чіткої інформації
emotion_labels = lbl_enc.classes_
report = classification_report(y_val, y_pred, target_names=emotion_labels, output_dict=True, zero_division=0)

# Виводимо метрики
for emotion in emotion_labels:
    print(f"{emotion}: Precision: {report[emotion]['precision']:.2f}, Recall: {report[emotion]['recall']:.2f}, F1-Score: {report[emotion]['f1-score']:.2f}")

# Час
print('Time: ', time.time() - start_time, 'seconds')
from nltk.tokenize import sent_tokenize

# Приклад
text = """I'm feeling great today! I finished all my work early and I'm going out for dinner with friends later.
However, I'm a bit nervous about my presentation tomorrow. But overall, I look forward for tomorrow!"""


nltk.download('punkt_tab')
# Розбиваємо текст на речення
sentences = sent_tokenize(text)

sentence_vectors = tfidf_vect.transform(sentences) # представляємо кожне речення у вигляді вектора, використовуючи TF-IDF

# Використовуючи наївний Баєсів класифікатор, передбачаємо ймовірність кожної емоції
sentence_emotions = nb.predict(sentence_vectors)

# Рахуємо ймовірності всіх емоцій для всіх речень
sentence_probabilities = nb.predict_proba(sentence_vectors)

# Рахуємо середнє квадратичне (з точністю до постійного множника) кожної емоції і беремо максимальну
weighted_emotion_scores = np.sqrt(np.sum(np.square(sentence_probabilities), axis=0))
final_emotion_idx = np.argmax(weighted_emotion_scores)

# Повертаємо емоцію (бо на попередньому кроці ми отримали індекс - мітку домінуючої емоції)
final_emotion = lbl_enc.inverse_transform([final_emotion_idx])[0]

# Для кожного речення виводимо домінуючу емоцію
print("Sentence-level emotions:")
for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}\nPredicted emotion: {lbl_enc.inverse_transform([sentence_emotions[i]])[0]}\n")

# Домінуюча емоція для всього тексту
print(f"\nOverall predicted emotion for the text: {final_emotion}")