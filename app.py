import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

import nltk
nltk.data.path.append("nltk_data")

# Try loading stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# ---------- Load model & vectorizer (with caching) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_artifacts()

# ---------- Text cleaning (same logic as notebook) ----------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)      # keep only letters + space
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_sentiment(review: str) -> str:
    clean = clean_text(review)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    return pred

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Mobile Review Sentiment", page_icon="📱", layout="centered")

st.title("📱 Mobile Review Sentiment Analyzer")
st.write("Predict whether a customer review is **Positive** or **Negative**.")

user_input = st.text_area(
    "Enter a mobile product review:",
    height=150,
    placeholder="Example: The battery life is great and the camera is amazing!"
)

if st.button("🔍 Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please type a review first.")
    else:
        label = predict_sentiment(user_input)

        if label == "positive":
            st.success("✅ Sentiment: **POSITIVE** 😊")
        else:
            st.error("❌ Sentiment: **NEGATIVE** 😠")

st.markdown("---")
st.caption("Model: TF-IDF + SMOTE + LinearSVC (binary sentiment)")
