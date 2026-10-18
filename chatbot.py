import os
import json
import pickle
import random
from nlp_utils import preprocess

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIDENCE_THRESHOLD = 0.30   # isse kam bharosa ho to fallback jawab

# Model aur intents load karo
with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    saved = pickle.load(f)
vectorizer = saved["vectorizer"]
model = saved["model"]

with open(os.path.join(BASE, "intents.json"), encoding="utf-8") as f:
    intents = json.load(f)["intents"]

responses = {i["tag"]: i["responses"] for i in intents}

FALLBACK = ("Sorry, I did not understand that. "
            "You can ask me about AI, machine learning, NLP or chatbots.")


def predict_intent(text):
    """Intent aur confidence return karta hai."""
    X = vectorizer.transform([preprocess(text)])
    probs = model.predict_proba(X)[0]
    best = probs.argmax()
    return model.classes_[best], float(probs[best])


def get_response(text):
    """(reply, intent, confidence) return karta hai."""
    intent, confidence = predict_intent(text)
    # kam bharosa ya scope se bahar ka sawal: fallback jawab
    if confidence < CONFIDENCE_THRESHOLD or intent == "out_of_scope":
        return FALLBACK, "unknown", confidence
    return random.choice(responses[intent]), intent, confidence


if __name__ == "__main__":
    print("Chatbot ready! Type 'quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit", "bye bye"):
            print("Bot: Goodbye!")
            break
        reply, intent, conf = get_response(user)
        print(f"Bot: {reply}   [intent={intent}, confidence={conf:.2f}]")