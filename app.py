import streamlit as st
from chatbot import chatbot
import uuid

# Unique config for chatbot
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
mybot = chatbot()
workflow = mybot() ####

# Set custom page config
st.set_page_config(page_title="AI ChatBot", page_icon="🤖", layout="centered")

# Apply custom CSS for a sleek UI with better readability
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #d3d3d3 
        }
        .stApp {
            background: #1e1e1e;
            padding: 20px;
        }
        .title {
            text-align: center;
            font-size: 38px;
            font-weight: bold;
            color: #00eaff;  /* Soft cyan for visibility */
            text-shadow: 2px 2px 12px rgba(0, 234, 255, 0.5);
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #d3d3d3;  /* Light gray for better readability */
        }
        .input-box {
            border-radius: 12px;
            padding: 12px;
            font-size: 18px;
            border: 2px solid #444;
            background-color: rgba(255, 255, 255, 0.1);
            color: #d3d3d3;
        }
        .chat-bubble {
            background: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 15px;
            margin-top: 10px;
            color: #f1f1f1;
            font-size: 18px;
            box-shadow: 2px 2px 12px rgba(255, 255, 255, 0.1);
        }
        .stTextInput > div > div > input {
            background-color: #252525;
            color: #d3d3d3;
            border-radius: 8px;
            padding: 10px;
            border: 2px solid #444;
        }
        .stButton button {
            background: linear-gradient(135deg, #ff8c00, #ff512f);
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 12px;
            font-weight: bold;
            border: none;
            transition: 0.3s ease-in-out;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #ff6b00, #ff002f);
            transform: scale(1.05);
            box-shadow: 0px 0px 15px rgba(255, 105, 135, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="title">🤖 AI ChatBot with LangGraph</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask me anything, and I will try to answer! 🧐</p>', unsafe_allow_html=True)

# User Input
question = st.text_input("Enter your question here:", key="input_question")

input_data = {"question": question}

# Generate response
if st.button("Get Answer 🎯"):
    if question.strip():
        with st.spinner("Thinking... 🤖"):
            response = workflow.invoke(input_data, config=config)
            st.markdown(f'<div class="chat-bubble"><b>Answer:</b> {response["response"]}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question before submitting.")
