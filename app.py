import streamlit as st
from chatbot import get_response

st.set_page_config(page_title="AI Chatbot using NLP", page_icon="🤖")

st.title("🤖 AI Chatbot using NLP")
st.caption("Built with Python, NLTK and scikit-learn | Vinay Petkar | AI and ML Internship, Cognevance")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This chatbot recognizes the intent of your message using TF-IDF and "
             "Logistic Regression, then replies with a suitable answer.")
    st.header("Try asking")
    st.write("- hello\n- what is machine learning\n- what is nlp\n- what is a chatbot\n"
             "- why python for ai\n- who made you\n- thank you")
    show_debug = st.checkbox("Show intent and confidence", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history ko yaad rakhna
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am an NLP chatbot. Ask me about AI, ML, NLP or chatbots.", "info": ""}
    ]

# Purani chat dikhao
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if show_debug and m.get("info"):
            st.caption(m["info"])

# Naya message
user_text = st.chat_input("Type your message here...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text, "info": ""})
    with st.chat_message("user"):
        st.write(user_text)

    reply, intent, conf = get_response(user_text)
    info = f"intent: {intent} | confidence: {conf:.2f}"
    st.session_state.messages.append({"role": "assistant", "content": reply, "info": info})
    with st.chat_message("assistant"):
        st.write(reply)
        if show_debug:
            st.caption(info)