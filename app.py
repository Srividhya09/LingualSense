import streamlit as st
import numpy as np
import re
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Backend File Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "gru" / "GRU_Model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.joblib"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.joblib"

# Must be the same max_length used during GRU training
MAX_LENGTH = 200


# ============================================================
# Load the Already-Trained GRU Backend
# ============================================================

@st.cache_resource
def load_backend():

    # Load the already-trained GRU model
    model = load_model(MODEL_PATH)

    # Load the tokenizer that was fitted during training
    tokenizer = joblib.load(TOKENIZER_PATH)

    # Load the label encoder used during training
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return model, tokenizer, label_encoder


# ============================================================
# Text Preprocessing
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)   # remove ALL punctuation (data has none)
    text = re.sub(r'\d+', '', text)
    return text.strip()


def preprocess_text(text, tokenizer):

    # Clean the user input
    cleaned_text = clean_text(text)

    # Convert text into the same integer representation
    # used during model training
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad the sequence to the same length used during training
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
    )

    return padded_sequence


# ============================================================
# Set Page Configuration for Better UI
# ============================================================

st.set_page_config(
    page_title="LingualSense - Your Multilingual Companion",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# Custom CSS for Styling
# ============================================================

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


# ============================================================
# Title and Subtitle
# ============================================================

st.markdown(
    "<div class='title-text'>LingualSense - Your Multilingual Companion 🌍</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle-text'>Effortlessly Detect Languages and Break Communication Barriers!</div>",
    unsafe_allow_html=True
)


# ============================================================
# Input Section
# ============================================================

st.markdown(
    "<h3 style='color: white;'>Enter Your Text Below</h3>",
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Paste or Type Your Text Here:",
    placeholder="Type in any language you like..."
)


# ============================================================
# Language Detection using Trained GRU Model
# ============================================================
if st.button("Detect Language"):

    if user_input.strip() == "":
        st.warning("Please enter some text to detect the language.")

    else:

        try:

            # Load the already-trained model and preprocessing artifacts
            model, tokenizer, label_encoder = load_backend()

            # Preprocess the new user input
            processed_input = preprocess_text(
                user_input,
                tokenizer
            )

            # Get prediction from the trained GRU
            prediction = model.predict(
                processed_input,
                verbose=0
            )

            # Find the class with the highest probability
            predicted_class = np.argmax(
                prediction[0]
            )

            # Convert predicted class number back to language name
            detected_language = label_encoder.inverse_transform(
                [predicted_class]
            )[0]

            # Display detected language
            st.markdown(
                f"<div class='language-result'>Detected Language: <strong>{detected_language.upper()}</strong></div>",
                unsafe_allow_html=True
            )

            # # ---------- TEMP DEBUG (delete later) ----------
            # probs = prediction[0]
            # top3 = np.argsort(probs)[::-1][:3]
            # for i in top3:
            #     st.write(label_encoder.inverse_transform([i])[0], f"{probs[i]*100:.1f}%")

            # seq = tokenizer.texts_to_sequences([clean_text(user_input)])[0]
            # st.write("OOV share:", f"{seq.count(tokenizer.word_index['<OOV>']) / max(len(seq),1) * 100:.1f}%")
            # # ------------------------------------------------

        except Exception as e:

            st.error(f"Error: {e}")

# ============================================================
# Footer
# ============================================================

st.markdown("""
    <hr>
    <p>
        Breaking Boundaries, Connecting Cultures 🌐 | Your Gateway to Multilingual Connections 🌎
    </p>
""", unsafe_allow_html=True)