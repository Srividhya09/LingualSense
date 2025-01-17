import streamlit as st
from langdetect import detect

# Set Page Configuration for Better UI
st.set_page_config(page_title="LingoDetect - Your Multilingual Companion", page_icon="🌍", layout="wide")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        .main {
            background-color: #6A1B9A;
            font-family: 'Arial', sans-serif;
        }
        .title-text {
            text-align: center;
            font-size: 2.8rem;
            color: #FFFFFF; /* Changed to white */
            font-weight: bold;
        }
        .subtitle-text {
            text-align: center;
            font-size: 1.2rem;
            color: white;
        }
        .stTextArea > label > div > span {
            font-size: 1rem;
            color: white;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 1rem;
            padding: 8px 16px;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .language-result {
            color: white;
            font-size: 1rem;
            text-align: center;
        }
        hr {
            border: 1px solid white;
        }
        p {
            color: white;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown("<div class='title-text'>LingoDetect - Your Multilingual Companion 🌍</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Effortlessly Detect Languages and Break Communication Barriers!</div>", unsafe_allow_html=True)

# Input Section
st.markdown("<h3 style='color: white;'>Enter Your Text Below</h3>", unsafe_allow_html=True)
user_input = st.text_area("Paste or Type Your Text Here:", placeholder="Type in any language you like...")

# Language Detection
if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text to detect the language.")
    else:
        try:
            detected_language = detect(user_input)
            st.markdown(f"<div class='language-result'>Detected Language: <strong>{detected_language.upper()}</strong></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error("Sorry, we couldn't detect the language. Please try again.")

# Footer
st.markdown("""
    <hr>
    <p>
        Breaking Boundaries, Connecting Cultures 🌐 | Your Gateway to Multilingual Connections 🌎
    </p>
""", unsafe_allow_html=True)
