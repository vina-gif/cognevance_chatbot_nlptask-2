# cognevance_chatbot_nlptask-2

# AI Chatbot using NLP

An intent-based chatbot built with Python and Natural Language Processing. It understands the intent of a user's message and replies with a suitable answer. A Streamlit web interface is included.

**Author:** Vinay Petkar | AI and ML Internship, Cognevance (Level 2)

## Features
- Text preprocessing with NLTK (lowercase, punctuation removal, tokenization, lemmatization)
- Intent recognition using TF-IDF and Logistic Regression (scikit-learn)
- Confidence threshold and out-of-scope detection with a fallback reply
- Chat interface built with Streamlit

## Project Structure
- `intents.json` : training dataset (intents, patterns and responses)
- `nlp_utils.py` : NLP preprocessing script
- `train_model.py` : trains the model and saves `model.pkl`
- `chatbot.py` : intent prediction and response generation (also runs in the terminal)
- `app.py` : Streamlit chat interface
- `test_chatbot.py` : accuracy test on 28 test sentences
- `test_results.txt` : test output
- `chat_demo.png`, `Fall_demo.png` : demo screenshots

## How to Run
```
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python train_model.py
streamlit run app.py
```
To chat in the terminal instead, run `python chatbot.py`.

## Testing and Results
The chatbot was tested on 28 sentences (24 in-scope, 4 out-of-scope).

| Round | Change | Accuracy |
|---|---|---|
| 1 | First model | 96.4% (27/28) |
| 2 | Added chatbot patterns | 92.9% |
| 3 | Added out_of_scope intent and more patterns | 96.4% |
| 4 | Confidence threshold 0.35 to 0.30 | 100% (28/28) |
| 5 | Added weather and joke examples after a manual test failure | 100% (28/28) |

This is a small development test set, so real-world accuracy may be lower.

## Tech Stack
Python, NLTK, scikit-learn, Streamlit
