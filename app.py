import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
spell = SpellChecker()


# =================== EXAMPLE REVIEWS ===================

POSITIVE_EXAMPLES = [
    "This phone is excellent, the camera quality is outstanding and battery lasts all day.",
    "Amazing performance, super smooth and fast. Totally worth the price!",
    "The display is bright and vibrant, and the speakers are surprisingly loud.",
    "I love this device, charging is very fast and there is no overheating.",
    "Great value for money, the phone feels premium and runs all apps easily.",
    "The camera is crystal clear and low-light photos are impressive.",
    "Battery backup is superb, easily lasts more than a day with heavy use.",
    "The UI is clean and user-friendly, no lags or issues so far.",
    "Awesome phone for gaming, graphics run smoothly without frame drops.",
    "Really happy with the purchase, build quality is top-notch and durable.",
]

NEGATIVE_EXAMPLES = [
    "Very disappointing phone, battery drains extremely fast and heats up a lot.",
    "The camera quality is terrible, pictures look blurry even in daylight.",
    "The phone keeps hanging and apps crash frequently, waste of money.",
    "Poor build quality, the back panel started making noise within a week.",
    "The display is dull and colors look washed out. Not satisfied at all.",
    "Charging is very slow and the phone gets hot even during normal use.",
    "Speakers are very weak and the call quality is also bad.",
    "Worst phone ever, performance is laggy and not suitable even for basic use.",
    "Network issues everywhere, calls drop and internet is unstable.",
    "The phone started showing problems within a month, bad experience overall.",
]

# =================== NLTK SETUP ===================

# Add custom nltk path for Streamlit Cloud and try loading stopwords
nltk.data.path.append("nltk_data")

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


# =================== LOAD MODEL & VECTORIZER ===================

@st.cache_resource
def load_artifacts():
    # Make sure these filenames match your repo exactly
    model = joblib.load("sentiment_svm_model.pkl")
    # If your file is named 'tfidf_vectorizer (1).pkl', change the line below:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    # vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
    return model, vectorizer


model, tfidf = load_artifacts()


# =================== PREPROCESSING & PREDICTION ===================

def clean_text(text: str) -> str:
    """Clean and lightly spell-correct the review text."""
    # Lowercase
    text = str(text).lower()
    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    # Tokenize
    words = text.split()

    cleaned_words = []
    for w in words:
        if w in stop_words:
            continue  # remove stopwords

        # If spelling is wrong, spell.correction will try to fix it
        corrected = spell.correction(w)
        if corrected is None:
            corrected = w

        cleaned_words.append(corrected)

    return " ".join(cleaned_words)



def predict_sentiment(review: str) -> str:
    """Return 'positive' or 'negative' using the trained SVM model."""
    clean = clean_text(review)
    vec = tfidf.transform([clean])
    label = model.predict(vec)[0]  # 'positive' or 'negative'
    return label, clean


# =================== STREAMLIT UI ===================

st.set_page_config(
    page_title="Mobile Review Sentiment Analyzer",
    page_icon="📱",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("ℹ️ About Project")
    st.markdown(
        """
        **NLP – Sentiment Analysis**

        - Domain: *Mobile product reviews*  
        - Task: Binary sentiment classification  
          (**Positive** vs **Negative**)  
        - Model: **TF-IDF + SMOTE + LinearSVC**  
        - Test accuracy: ~**91%**  
        
        **Pipeline:**
        1. Rating → sentiment labels  
        2. Text cleaning (lowercase, stopword removal)  
        3. TF-IDF (uni + bi-grams)  
        4. SMOTE for class balancing  
        5. LinearSVC training
        """
    )
    st.markdown("---")
    st.markdown("🔹 Type your own review or use example buttons to auto-fill text.")


# ---------- Main Title ----------
st.markdown("## 📱 Mobile Review Sentiment Analyzer")
st.write(
    "Enter a customer review about a mobile phone and the model will predict "
    "whether the sentiment is **Positive** or **Negative**."
)

# ---------- Session State Init ----------
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

if "pos_idx" not in st.session_state:
    st.session_state["pos_idx"] = 0

if "neg_idx" not in st.session_state:
    st.session_state["neg_idx"] = 0

# ---------- Example Buttons ----------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔹 Positive Example"):
        idx = st.session_state["pos_idx"]
        st.session_state["review_text"] = POSITIVE_EXAMPLES[idx]
        st.session_state["pos_idx"] = (idx + 1) % len(POSITIVE_EXAMPLES)

with col2:
    if st.button("🔸 Negative Example"):
        idx = st.session_state["neg_idx"]
        st.session_state["review_text"] = NEGATIVE_EXAMPLES[idx]
        st.session_state["neg_idx"] = (idx + 1) % len(NEGATIVE_EXAMPLES)

with col3:
    if st.button("🧹 Clear Text"):
        st.session_state["review_text"] = ""

# ---------- Text Area ----------
user_input = st.text_area(
    "Enter a mobile product review:",
    key="review_text",
    height=150,
    placeholder="Example: The battery life is great and the camera is amazing!",
)

# ---------- Predict Button ----------
if st.button("🔍 Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please type or choose a review first.")
    else:
        label, cleaned = predict_sentiment(user_input)

        if label == "positive":
            st.success("✅ Sentiment: **POSITIVE** 😊")
        else:
            st.error("❌ Sentiment: **NEGATIVE** 😠")

        with st.expander("🔎 View preprocessed (cleaned) text"):
            st.code(cleaned, language="text")

st.markdown("---")
st.caption("Backend: TF-IDF (uni+bi-grams) + SMOTE + LinearSVC – Binary Sentiment Model")
