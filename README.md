# LingualSense 🌍 - Multilingual Language Detection

LingualSense identifies the language of a piece of text using a deep learning model trained from scratch. A character-level GRU network recognizes **28 languages** and is served through a simple Streamlit web app.

---

## Features

- Detects 28 languages, including languages with no spaces between words (Chinese, Japanese, Thai)
- Character-level tokenization, so it handles short text, names and unseen words
- Compares three architectures (Neural Network, LSTM, GRU) and uses the best one
- Shows the detected language with a confidence score
- Trained and evaluated in Google Colab, deployed with Streamlit

## Supported Languages

Arabic, Chinese, Danish, Dutch, English, Estonian, French, German, Greek, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Latin, Malayalam, Persian, Portuguese, Pushto, Romanian, Russian, Spanish, Swedish, Tamil, Thai, Turkish, Urdu.

---

## How It Works

1. **Data cleaning:** remove duplicate rows, lowercase the text, strip punctuation and numbers, and merge misspelled duplicate labels (Portugese/Portugeese into Portuguese, Sweedish into Swedish).
2. **Tokenization:** each text becomes a sequence of characters (character-level tokenizer with an OOV token), padded to 200 characters.
3. **Models:** three classifiers are trained on the same 80/20 split:
   - Baseline Neural Network (Embedding, Flatten, Dense)
   - LSTM (Embedding, LSTM, Dense)
   - GRU (Embedding, GRU, Dense)
4. **Prediction:** the app cleans the input the same way, runs the GRU model and returns the most likely language.

## Model Comparison

All models were trained for 10 epochs on the same data (25,704 training rows and 6,426 test rows).

| Model | Accuracy | Macro F1 | Worst-language F1 | Parameters | Size |
|-------|----------|----------|-------------------|------------|------|
| Neural Network | 91.19% | 0.908 | 0.566 | 2.49M | 29.9 MB |
| LSTM | 95.24% | 0.953 | 0.836 | 0.99M | 11.9 MB |
| **GRU** | **96.70%** | **0.967** | **0.891** | 0.96M | 11.5 MB |

Accuracy on text cut to its first N characters (short-text robustness):

| Model | 20 chars | 50 chars | 100 chars |
|-------|----------|----------|-----------|
| Neural Network | 48.9% | 70.2% | 84.0% |
| LSTM | 68.4% | 84.8% | 92.4% |
| **GRU** | **70.8%** | **87.2%** | **94.4%** |

**The GRU was chosen** because it had the best accuracy and macro F1, the strongest results on short text, and the smallest model. The Neural Network overfitted: its validation loss rose while its training loss kept falling.

---

## Run Locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run main.py
```

The model was trained with TensorFlow 2.20.0 and Keras 3.13.2. Use the same versions to avoid loading errors.

## Retrain the Models

Open `LingualSense.ipynb` in Google Colab (a GPU runtime is recommended), upload `merged_dataset.csv`, and run all cells. The notebook saves the tokenizer, label encoder and all three models.

---

## Limitations

- The model knows only the 28 languages above. For any other language it still returns the closest one of them, because it has no "unknown" option.
- Very short input (a word or two) is less reliable.
- Each input gets a single language, so mixed-language text returns the dominant one.
- The model learned from the training data's style (mostly full sentences). Slang and unusual text may be less accurate.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Streamlit, Google Colab

## Author

_Chakilela Srividhya_
