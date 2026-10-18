from chatbot import get_response

# (sawal, expected intent)  -- ye sentences training data mein nahi hain
tests = [
    ("hey there", "greeting"),
    ("good afternoon", "greeting"),
    ("hi bot", "greeting"),
    ("see you tomorrow", "goodbye"),
    ("bye for now", "goodbye"),
    ("thanks for your help", "thanks"),
    ("thank you very much", "thanks"),
    ("what are you called", "name"),
    ("who are you", "name"),
    ("what can you help me with", "capabilities"),
    ("what is artificial intelligence", "ai_definition"),
    ("explain ai to me", "ai_definition"),
    ("what is ml", "ml_definition"),
    ("how does machine learning work", "ml_definition"),
    ("what is natural language processing", "nlp_definition"),
    ("explain nlp", "nlp_definition"),
    ("what is a chatbot", "chatbot_definition"),
    ("how do chatbots work", "chatbot_definition"),
    ("what is deep learning", "deep_learning"),
    ("explain neural network", "deep_learning"),
    ("why is python good for ai", "python_info"),
    ("which language is best for ai", "python_info"),
    ("who built you", "internship"),
    ("what is this project about", "internship"),
    # Ye sawal bot ke scope se bahar hain, jawab "unknown" hona chahiye
    ("how is the weather", "unknown"),
    ("tell me a joke", "unknown"),
    ("what is the capital of france", "unknown"),
    ("play some music", "unknown"),
]

correct = 0
wrong = []
for text, expected in tests:
    _, predicted, conf = get_response(text)
    ok = predicted == expected
    correct += ok
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {text!r:45} expected={expected:18} got={predicted:18} conf={conf:.2f}")
    if not ok:
        wrong.append((text, expected, predicted))

print("\n==============================")
print(f"Correct: {correct}/{len(tests)}")
print(f"Accuracy: {correct / len(tests) * 100:.1f}%")