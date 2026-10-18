import re
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def preprocess(text):
    """Text ko clean karke model ke liye ready karta hai."""
    text = text.lower()                          # 1. chhote akshar
    text = re.sub(r"[^a-z0-9\s]", "", text)      # 2. punctuation hatao
    tokens = nltk.word_tokenize(text)            # 3. shabdon mein todo
    tokens = [lemmatizer.lemmatize(t) for t in tokens]   # 4. root form
    return " ".join(tokens)


if __name__ == "__main__":
    print(preprocess("Hello!! What are you doing?"))