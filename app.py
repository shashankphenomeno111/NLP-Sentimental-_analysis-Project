import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# ----------------------- NLTK SETUP -----------------------
# Try to load stopwords, download if missing
nltk.data.path.append("nltk_data")
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ----------------------- LOAD MODEL -----------------------
@st.cache_resource
def load_artifacts():
    # NOTE: make sure these names match the files in your repo
    model = joblib.load("sentiment_svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # or "tfidf_vectorizer (1).pkl"
    return model, vectorizer

model, tfidf = load_artifacts()

# ----------------------- PREPROCESSING -----------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)      # keep only letters + space
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# OPTIONAL: tiny rule-based neutral detector (for display only)
NEUTRAL_KEYWORDS = ["okay", "ok", "average", "fine", "decent", "not bad", "not good"]

def rule_based_neutral(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in NEUTRAL_KEYWORDS)

def predict_sentiment(review: str) -> str:
    """
    Returns: 'positive', 'negative' OR 'neutral' (heuristic).
    The model itself is binary (pos/neg); neutral is rule-based for display.
    """
    # If strongly neutral-sounding, mark as neutral first
    if rule_based_neutral(review):
        return "neutral"

    clean = clean_text(review)
    vec = tfidf.transform([clean])
    label = model.predict(vec)[0]  # 'positive' or 'negative'
    return label

# ----------------------- STREAMLIT UI -----------------------
st.set_page_config(
    page_title="Mobile Review Sentiment Analyzer",
    page_icon="📱",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("ℹ️ About Project")
    st.markdown(
        """
        **NLP – Sentiment Analysis**

        - Domain: *Mobile product reviews*  
        - Goal: Classify reviews as **Positive** or **Negative**  
        - Model: **TF-IDF + SMOTE + LinearSVC**  
        - Train accuracy: ~91%  
        - Techniques:
          - Text cleaning (lowercase, stopword removal)
          - TF-IDF (uni + bi-grams)
          - SMOTE for class balancing
        """
    )
    st.markdown("---")
    st.markdown("**How to use:**\n1. Type or paste a review.\n2. Click **Predict Sentiment**.\n3. See the output and cleaned text.")

# Main title
st.markdown("## 📱 Mobile Review Sentiment Analyzer")
st.write("Predict whether a customer review is **Positive** or **Negative** based on the text.")

# Example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🔹 Example Positive Review"):
        st.session_state["example_text"] = "This phone is excellent, battery life is great and camera is amazing!"
with col2:
    if st.button("🔸 Example Negative Review"):
        st.session_state["example_text"] = "Worst phone ever, battery drains fast and the screen quality is terrible."

default_text = st.session_state.get("example_text", "")

# Input area
user_input = st.text_area(
    "Enter a mobile product review:",
    value=default_text,
    height=150,
    placeholder="Example: The battery life is great and the camera is amazing!"
)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please type or select a review first.")
    else:
        label = predict_sentiment(user_input)
        cleaned = clean_text(user_input)

        if label == "positive":
            st.success("✅ **Sentiment: POSITIVE** 😊")
        elif label == "negative":
            st.error("❌ **Sentiment: NEGATIVE** 😠")
        else:  # neutral (rule-based)
            st.info("😐 **Sentiment: NEUTRAL (rule-based)**")

        # Show cleaned text for explanation
        with st.expander("🔎 See preprocessed (cleaned) text"):
            st.code(cleaned, language="text")

st.markdown("---")
st.caption("Backend: TF-IDF (uni+bi-grams) + SMOTE + LinearSVC (binary sentiment model)")
