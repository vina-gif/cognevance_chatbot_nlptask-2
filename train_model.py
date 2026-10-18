import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nlp_utils import preprocess

BASE = os.path.dirname(os.path.abspath(__file__))

# 1. Training data load karo
with open(os.path.join(BASE, "intents.json"), encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(preprocess(pattern))
        labels.append(intent["tag"])

print("Total training sentences:", len(texts))
print("Total intents:", len(set(labels)))

# 2. Accuracy check: data ko train/test mein baanto
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
vec = TfidfVectorizer(ngram_range=(1, 2))
model = LogisticRegression(C=10, max_iter=1000)
model.fit(vec.fit_transform(X_train), y_train)
acc = accuracy_score(y_test, model.predict(vec.transform(X_test)))
print(f"Test accuracy: {acc * 100:.1f}%")

# 3. Final model: poore data par train karke save karo
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)
final_model = LogisticRegression(C=10, max_iter=1000)
final_model.fit(X, labels)

with open(os.path.join(BASE, "model.pkl"), "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "model": final_model}, f)

print("Model trained and saved as model.pkl")
