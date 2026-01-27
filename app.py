"""
📱 NLP Sentiment Analysis Dashboard
====================================
A creative multi-page dashboard for mobile review sentiment analysis.
Built with Streamlit, featuring comprehensive EDA, model insights, and real-time predictions.

Author: Shashank R
Model: TF-IDF + SMOTE + LinearSVC (91% Accuracy)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize spell checker
spell = SpellChecker()

# =================== PAGE CONFIG ===================

st.set_page_config(
    page_title="📱 NLP Sentiment Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== CUSTOM CSS - ULTRA DARK THEME ===================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Theme - Ultra Dark Gradient */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 40%, #0f0f1f 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121f 0%, #0d0d1a 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    /* Radio button styling for navigation */
    [data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(102, 126, 234, 0.1);
        padding: 12px 20px;
        border-radius: 12px;
        margin: 3px 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
        color: #e2e8f0 !important;
        font-size: 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(102, 126, 234, 0.3);
        transform: translateX(5px);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-color: transparent;
    }
    
    /* Glassmorphism Cards - Dark Glass */
    .glass-card {
        background: rgba(20, 20, 40, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric Cards - More Vibrant */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25), rgba(118, 75, 162, 0.25));
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 120px;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    .metric-label {
        color: #cbd5e0;
        font-size: 1rem;
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Headers - Brighter */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    h1 {
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Description Boxes - More Visible */
    .description-box {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1));
        border-left: 4px solid #818cf8;
        padding: 18px 22px;
        border-radius: 0 15px 15px 0;
        margin: 12px 0;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.7;
    }
    
    /* Success/Error Boxes - More Vibrant */
    .success-prediction {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.25), rgba(22, 163, 74, 0.2));
        border: 2px solid #22c55e;
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(34, 197, 94, 0.3);
    }
    
    .error-prediction {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(220, 38, 38, 0.2));
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.3);
    }
    
    /* Button Styling - More Prominent */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Text Area - Better Visibility */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 15px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        padding: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Expander - Brighter */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(0, 0, 0, 0.2);
        border-radius: 20px 20px 0 0;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.4); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.6); }
    }
    
    .animate-fade {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .animate-glow {
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Stats Grid */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 25px 0;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.2);
    }
    
    /* Navigation Active State */
    .nav-active {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    /* Plotly Chart Background Fix */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    /* Metric override for better visibility */
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e0 !important;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# =================== NLTK SETUP ===================

nltk.data.path.append("nltk_data")

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# =================== LOAD DATA & MODEL ===================

@st.cache_data
def load_data():
    """Load and preprocess the dataset for EDA."""
    df = pd.read_csv("dataset -P582 (1).csv")
    df['sentiment'] = df['rating'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')
    df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    return df

@st.cache_resource
def load_artifacts():
    """Load the trained model and vectorizer."""
    model = joblib.load("sentiment_svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# Load data and model
df = load_data()
model, tfidf = load_artifacts()

# =================== EXAMPLE REVIEWS ===================

POSITIVE_EXAMPLES = [
    "This phone is excellent, the camera quality is outstanding and battery lasts all day.",
    "Amazing performance, super smooth and fast. Totally worth the price!",
    "The display is bright and vibrant, and the speakers are surprisingly loud.",
    "I love this device, charging is very fast and there is no overheating.",
    "Great value for money, the phone feels premium and runs all apps easily.",
    "Best phone I have ever used, the features are incredible!",
    "Fantastic build quality and the software is very smooth.",
    "Highly recommend this phone, exceeded all my expectations!",
]

NEGATIVE_EXAMPLES = [
    "Very disappointing phone, battery drains extremely fast and heats up a lot.",
    "The camera quality is terrible, pictures look blurry even in daylight.",
    "The phone keeps hanging and apps crash frequently, waste of money.",
    "Poor build quality, the back panel started making noise within a week.",
    "Worst phone ever, performance is laggy and not suitable even for basic use.",
    "Do not buy this phone, it stopped working after just one month.",
    "Terrible customer service and the phone has so many bugs.",
    "Complete waste of money, returning this immediately.",
]

# =================== HELPER FUNCTIONS ===================

def clean_text(text: str) -> str:
    """Clean and preprocess the review text."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    cleaned_words = []
    for w in words:
        if w in stop_words:
            continue
        corrected = spell.correction(w)
        if corrected is None:
            corrected = w
        cleaned_words.append(corrected)
    return " ".join(cleaned_words)

def predict_sentiment(review: str):
    """Predict sentiment with confidence score."""
    clean = clean_text(review)
    vec = tfidf.transform([clean])
    prediction = model.predict(vec)[0]
    try:
        confidence = abs(model.decision_function(vec)[0])
        confidence = min(confidence / 2, 1.0)
    except:
        confidence = 0.91
    return prediction, clean, confidence

def create_wordcloud(texts, colormap='viridis'):
    """Generate a word cloud from texts."""
    all_text = ' '.join(texts)
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='rgba(0,0,0,0)',
        mode='RGBA',
        colormap=colormap,
        max_words=100
    ).generate(all_text)
    return wordcloud

def get_word_frequencies(texts, n=15):
    """Get top n word frequencies."""
    all_words = ' '.join(texts).lower().split()
    filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
    return Counter(filtered_words).most_common(n)

# =================== SESSION STATE ===================

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "pos_idx" not in st.session_state:
    st.session_state.pos_idx = 0
if "neg_idx" not in st.session_state:
    st.session_state.neg_idx = 0
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =================== SIDEBAR NAVIGATION ===================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 10px;'>
        <h1 style='font-size: 2.2rem; background: linear-gradient(90deg, #a78bfa, #818cf8); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin-bottom: 5px;'>
            📱 NLP Dashboard
        </h1>
        <p style='color: #94a3b8; font-size: 0.9rem;'>Sentiment Analysis System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Menu
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select a page:",
        ["🏠 Home", "📊 EDA Analysis", "🤖 Model Info", "🔍 Prediction"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats in Sidebar
    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📝 Reviews", f"{len(df):,}")
    with col2:
        st.metric("🎯 Accuracy", "91%")
    
    pos_count = len(df[df['sentiment'] == 'Positive'])
    neg_count = len(df[df['sentiment'] == 'Negative'])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("😊 Positive", f"{pos_count}")
    with col2:
        st.metric("😠 Negative", f"{neg_count}")
    
    st.markdown("---")
    
    # Tech Stack
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    <div style='color: #94a3b8; line-height: 2;'>
        • Python 3.9+<br>
        • Streamlit<br>
        • Scikit-Learn<br>
        • NLTK / Plotly<br>
        • LinearSVC Model
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Author
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <p style='color: #64748b; font-size: 0.85rem;'>
            Built by <strong style='color: #a78bfa;'>Shashank R</strong><br>
            Data Science Portfolio
        </p>
    </div>
    """, unsafe_allow_html=True)

# =================== PAGE: HOME ===================

if page == "🏠 Home":
    # Header with animation
    st.markdown("""
    <div class='animate-fade' style='text-align: center; padding: 50px 20px;'>
        <div style='font-size: 5rem; margin-bottom: 15px;'>📱</div>
        <h1 style='font-size: 3.2rem; margin-bottom: 15px; 
                   background: linear-gradient(90deg, #a78bfa, #818cf8, #667eea);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Mobile Review Sentiment Analyzer
        </h1>
        <p style='font-size: 1.3rem; color: #94a3b8; max-width: 700px; margin: auto;'>
            An intelligent NLP system powered by Machine Learning to analyze customer sentiment
            from mobile product reviews with <strong style='color: #22c55e;'>91% accuracy</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    pos_pct = (df['sentiment'] == 'Positive').mean() * 100
    neg_pct = (df['sentiment'] == 'Negative').mean() * 100
    
    with col1:
        st.markdown(f"""
        <div class='metric-card animate-fade'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>📝 Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card animate-fade'>
            <div class='metric-value'>{pos_pct:.0f}%</div>
            <div class='metric-label'>😊 Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card animate-fade'>
            <div class='metric-value'>{neg_pct:.0f}%</div>
            <div class='metric-label'>😠 Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card animate-fade animate-glow'>
            <div class='metric-value'>91%</div>
            <div class='metric-label'>🎯 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    <div class='glass-card animate-fade'>
        <h2>🎯 Project Overview</h2>
        <div class='description-box'>
            <strong>Objective:</strong> Build an accurate sentiment classification system 
            for mobile product reviews to help businesses understand customer opinions and improve products.
        </div>
        <p style='color: #e2e8f0; line-height: 1.9; font-size: 1.05rem;'>
            This project implements a complete <strong style='color: #a78bfa;'>Natural Language Processing (NLP) pipeline</strong> 
            that processes raw customer reviews and classifies them as either <strong style='color: #22c55e;'>Positive</strong> 
            or <strong style='color: #ef4444;'>Negative</strong> sentiment. The system leverages advanced text preprocessing, 
            TF-IDF feature extraction, SMOTE class balancing, and LinearSVC classification.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='color: #a78bfa;'>✨ Key Features</h3>
            <ul style='color: #e2e8f0; line-height: 2.2; font-size: 1.05rem;'>
                <li>🔤 <strong>Smart Text Preprocessing</strong> - Cleaning & spell correction</li>
                <li>📊 <strong>TF-IDF Vectorization</strong> - Uni-gram + Bi-gram features</li>
                <li>⚖️ <strong>SMOTE Balancing</strong> - Handles class imbalance</li>
                <li>🤖 <strong>LinearSVC Model</strong> - Fast & accurate classification</li>
                <li>⚡ <strong>Real-time Predictions</strong> - Instant sentiment analysis</li>
                <li>📈 <strong>Interactive EDA</strong> - Comprehensive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='color: #818cf8;'>🔄 ML Pipeline</h3>
            <ul style='color: #e2e8f0; line-height: 2.2; font-size: 1.05rem;'>
                <li>1️⃣ <strong>Data Collection</strong> - E-commerce mobile reviews</li>
                <li>2️⃣ <strong>Preprocessing</strong> - Clean & normalize text</li>
                <li>3️⃣ <strong>Feature Engineering</strong> - TF-IDF vectorization</li>
                <li>4️⃣ <strong>Class Balancing</strong> - SMOTE oversampling</li>
                <li>5️⃣ <strong>Model Training</strong> - LinearSVC classifier</li>
                <li>6️⃣ <strong>Deployment</strong> - Streamlit web application</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # How to Use
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <h3 style='color: #667eea;'>📖 How to Use This Dashboard</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 20px;'>
            <div class='feature-card'>
                <h4 style='color: #a78bfa; margin-bottom: 10px;'>📊 EDA Analysis</h4>
                <p style='color: #94a3b8;'>Explore data visualizations, rating distributions, 
                word clouds, and sentiment patterns with detailed explanations.</p>
            </div>
            <div class='feature-card'>
                <h4 style='color: #818cf8; margin-bottom: 10px;'>🤖 Model Info</h4>
                <p style='color: #94a3b8;'>Learn about the ML architecture, 
                performance metrics, confusion matrix, and feature importance.</p>
            </div>
            <div class='feature-card'>
                <h4 style='color: #22c55e; margin-bottom: 10px;'>🔍 Prediction</h4>
                <p style='color: #94a3b8;'>Enter any mobile review and get 
                instant sentiment prediction with confidence scores.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =================== PAGE: EDA ===================

elif page == "📊 EDA Analysis":
    st.markdown("""
    <div class='animate-fade' style='text-align: center; padding: 30px;'>
        <h1 style='font-size: 2.8rem;'>📊 Exploratory Data Analysis</h1>
        <p style='color: #94a3b8; font-size: 1.1rem;'>Deep dive into the mobile review dataset with comprehensive visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown("""
    <div class='glass-card'>
        <h2>📋 Dataset Overview</h2>
        <div class='description-box'>
            <strong>Description:</strong> This dataset contains customer reviews from e-commerce platforms (Amazon) 
            for various mobile phones. Each review includes a title, review text, and a rating from 1-5 stars.
            Reviews with ratings ≥4 are classified as <strong style='color: #22c55e;'>Positive</strong>, 
            while ratings ≤3 are classified as <strong style='color: #ef4444;'>Negative</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Samples", f"{len(df):,}")
    with col2:
        st.metric("📊 Features", "3")
    with col3:
        st.metric("🎯 Target Classes", "2")
    with col4:
        st.metric("📏 Avg. Words", f"{df['word_count'].mean():.0f}")
    
    with st.expander("🔍 View Sample Data", expanded=False):
        st.dataframe(df[['title', 'rating', 'sentiment']].head(10), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Rating Distribution
    st.markdown("""
    <div class='glass-card'>
        <h2>⭐ Rating Distribution</h2>
        <div class='description-box'>
            <strong>Description:</strong> This visualization shows how customers have rated mobile products.
            Understanding rating distribution helps identify customer satisfaction patterns and potential class imbalance.
            Higher ratings (4-5★) indicate satisfied customers, lower ratings (1-2★) indicate dissatisfaction.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rating_counts = df['rating'].value_counts().sort_index()
    fig_rating = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating (Stars)', 'y': 'Number of Reviews'},
        color=rating_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_rating.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        xaxis=dict(tickmode='linear', title_font_size=14),
        yaxis=dict(title_font_size=14),
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_rating, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"📈 **Most Common Rating:** {rating_counts.idxmax()}★ ({rating_counts.max()} reviews)")
    with col2:
        st.warning(f"📉 **Least Common Rating:** {rating_counts.idxmin()}★ ({rating_counts.min()} reviews)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sentiment Distribution
    st.markdown("""
    <div class='glass-card'>
        <h2>😊 Sentiment Distribution</h2>
        <div class='description-box'>
            <strong>Description:</strong> This pie chart shows the proportion of Positive vs Negative sentiments.
            A balanced dataset ensures the model learns both patterns effectively. SMOTE is used to handle any imbalance.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sentiment_counts = df['sentiment'].value_counts()
    fig_sentiment = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map={'Positive': '#22c55e', 'Negative': '#ef4444'},
        hole=0.45
    )
    fig_sentiment.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        height=400
    )
    fig_sentiment.update_traces(textinfo='percent+label', textfont_size=16)
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Text Length Analysis
    st.markdown("""
    <div class='glass-card'>
        <h2>📝 Text Length Analysis</h2>
        <div class='description-box'>
            <strong>Description:</strong> Understanding text length distribution helps in feature engineering.
            Very short reviews may lack context, while very long ones might contain noise.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📏 Mean Length", f"{df['text_length'].mean():.0f} chars")
    with col2:
        st.metric("📊 Median", f"{df['text_length'].median():.0f} chars")
    with col3:
        st.metric("📈 Maximum", f"{df['text_length'].max()} chars")
    with col4:
        st.metric("💬 Avg Words", f"{df['word_count'].mean():.0f}")
    
    fig_length = px.histogram(
        df, x='text_length', nbins=50,
        color='sentiment',
        color_discrete_map={'Positive': '#22c55e', 'Negative': '#ef4444'},
        labels={'text_length': 'Character Count', 'count': 'Frequency'},
        barmode='overlay',
        opacity=0.75
    )
    fig_length.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        height=400
    )
    st.plotly_chart(fig_length, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Word Clouds
    st.markdown("""
    <div class='glass-card'>
        <h2>☁️ Word Cloud Visualizations</h2>
        <div class='description-box'>
            <strong>Description:</strong> Word clouds show the most frequently used words in reviews.
            Larger words appear more often. Compare positive and negative reviews to see distinct vocabulary patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 😊 Positive Reviews")
        pos_texts = df[df['sentiment'] == 'Positive']['text'].tolist()
        pos_wc = create_wordcloud(pos_texts, 'Greens')
        fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
        ax_pos.imshow(pos_wc, interpolation='bilinear')
        ax_pos.axis('off')
        fig_pos.patch.set_alpha(0)
        st.pyplot(fig_pos)
        plt.close()
    
    with col2:
        st.markdown("### 😠 Negative Reviews")
        neg_texts = df[df['sentiment'] == 'Negative']['text'].tolist()
        neg_wc = create_wordcloud(neg_texts, 'Reds')
        fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
        ax_neg.imshow(neg_wc, interpolation='bilinear')
        ax_neg.axis('off')
        fig_neg.patch.set_alpha(0)
        st.pyplot(fig_neg)
        plt.close()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top Words
    st.markdown("""
    <div class='glass-card'>
        <h2>📊 Top Words by Sentiment</h2>
        <div class='description-box'>
            <strong>Description:</strong> These charts show the 15 most frequent words for each sentiment class.
            Notice how positive reviews focus on quality aspects while negative reviews emphasize problems.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Top Positive Words")
        pos_words = get_word_frequencies(pos_texts)
        pos_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
        fig_pos_bar = px.bar(pos_df, x='Count', y='Word', orientation='h',
                             color='Count', color_continuous_scale='Greens')
        fig_pos_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                   font_color='#e2e8f0', showlegend=False,
                                   yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_pos_bar, use_container_width=True)
    
    with col2:
        st.markdown("### 🔴 Top Negative Words")
        neg_words = get_word_frequencies(neg_texts)
        neg_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
        fig_neg_bar = px.bar(neg_df, x='Count', y='Word', orientation='h',
                             color='Count', color_continuous_scale='Reds')
        fig_neg_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                   font_color='#e2e8f0', showlegend=False,
                                   yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_neg_bar, use_container_width=True)

# =================== PAGE: MODEL INFO ===================

elif page == "🤖 Model Info":
    st.markdown("""
    <div class='animate-fade' style='text-align: center; padding: 30px;'>
        <h1 style='font-size: 2.8rem;'>🤖 Model Architecture & Performance</h1>
        <p style='color: #94a3b8; font-size: 1.1rem;'>Understanding the machine learning pipeline and evaluation metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Architecture
    st.markdown("""
    <div class='glass-card'>
        <h2>🏗️ Model Architecture</h2>
        <div class='description-box'>
            <strong>Description:</strong> Our sentiment analysis system uses a classical ML pipeline combining 
            TF-IDF vectorization, SMOTE balancing, and LinearSVC classification to achieve 91% accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #a78bfa;'>📊 TF-IDF</h3>
            <p style='color: #cbd5e0; font-size: 0.95rem; margin-top: 10px;'>
                <strong>Term Frequency-Inverse Document Frequency</strong><br><br>
                Converts text to numerical features by measuring word importance across documents.<br><br>
                • N-grams: (1, 2)<br>
                • Max Features: 5000
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #818cf8;'>⚖️ SMOTE</h3>
            <p style='color: #cbd5e0; font-size: 0.95rem; margin-top: 10px;'>
                <strong>Synthetic Minority Over-sampling</strong><br><br>
                Generates synthetic samples to balance class distribution.<br><br>
                • Prevents bias<br>
                • Improves recall
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #22c55e;'>🎯 LinearSVC</h3>
            <p style='color: #cbd5e0; font-size: 0.95rem; margin-top: 10px;'>
                <strong>Linear Support Vector Classifier</strong><br><br>
                Finds optimal hyperplane to separate sentiment classes.<br><br>
                • Fast inference<br>
                • High accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Pipeline Flow
    st.markdown("""
    <div class='glass-card'>
        <h2>🔄 Processing Pipeline</h2>
        <div class='description-box'>
            <strong>Description:</strong> The complete end-to-end pipeline from raw text input to sentiment prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 10px; padding: 20px;'>
        <div style='background: linear-gradient(135deg, #a78bfa, #818cf8); padding: 15px 25px; border-radius: 12px; color: white; font-weight: 600;'>📝 Raw Text</div>
        <span style='color: #818cf8; font-size: 1.5rem;'>→</span>
        <div style='background: linear-gradient(135deg, #818cf8, #667eea); padding: 15px 25px; border-radius: 12px; color: white; font-weight: 600;'>🔤 Lowercase</div>
        <span style='color: #667eea; font-size: 1.5rem;'>→</span>
        <div style='background: linear-gradient(135deg, #667eea, #6366f1); padding: 15px 25px; border-radius: 12px; color: white; font-weight: 600;'>🧹 Clean Text</div>
        <span style='color: #6366f1; font-size: 1.5rem;'>→</span>
        <div style='background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 15px 25px; border-radius: 12px; color: white; font-weight: 600;'>📊 TF-IDF</div>
        <span style='color: #8b5cf6; font-size: 1.5rem;'>→</span>
        <div style='background: linear-gradient(135deg, #8b5cf6, #a855f7); padding: 15px 25px; border-radius: 12px; color: white; font-weight: 600;'>🤖 LinearSVC</div>
        <span style='color: #a855f7; font-size: 1.5rem;'>→</span>
        <div style='background: linear-gradient(135deg, #22c55e, #16a34a); padding: 15px 25px; border-radius: 12px; color: white; font-weight: 600;'>✅ Sentiment</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown("""
    <div class='glass-card'>
        <h2>📈 Performance Metrics</h2>
        <div class='description-box'>
            <strong>Description:</strong> Model evaluation metrics showing accuracy, precision, recall, and F1-score.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card animate-glow'>
            <div class='metric-value'>91%</div>
            <div class='metric-label'>🎯 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.91</div>
            <div class='metric-label'>📊 Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.90</div>
            <div class='metric-label'>🔍 Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.90</div>
            <div class='metric-label'>⚖️ F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confusion Matrix
    st.markdown("""
    <div class='glass-card'>
        <h2>📊 Confusion Matrix</h2>
        <div class='description-box'>
            <strong>Description:</strong> Visualizes correct predictions (diagonal) vs misclassifications (off-diagonal).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    cm = np.array([[85, 9], [10, 96]])
    fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Negative', 'Positive'], y=['Negative', 'Positive'],
                       color_continuous_scale='Purples', text_auto=True)
    fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#e2e8f0', height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📋 Classification Report")
    report_df = pd.DataFrame({
        'Class': ['Negative', 'Positive', 'Weighted Avg'],
        'Precision': [0.89, 0.91, 0.91],
        'Recall': [0.90, 0.91, 0.90],
        'F1-Score': [0.90, 0.91, 0.90],
        'Support': [94, 106, 200]
    })
    st.dataframe(report_df.set_index('Class'), use_container_width=True)

# =================== PAGE: PREDICTION ===================

elif page == "🔍 Prediction":
    st.markdown("""
    <div class='animate-fade' style='text-align: center; padding: 30px;'>
        <h1 style='font-size: 2.8rem;'>🔍 Real-Time Sentiment Prediction</h1>
        <p style='color: #94a3b8; font-size: 1.1rem;'>Enter a mobile product review and get instant sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class='glass-card'>
        <h3>📖 How to Use</h3>
        <div class='description-box'>
            <strong>Instructions:</strong> Click an example button to auto-fill a sample review, 
            or type your own. Then click <strong>Predict</strong> to analyze the sentiment.
            Each button click shows a <strong>different example</strong> from our collection!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Example counter display
    st.markdown(f"""
    <div style='display: flex; justify-content: center; gap: 40px; margin-bottom: 20px;'>
        <div style='text-align: center; padding: 15px 30px; background: rgba(34, 197, 94, 0.2); 
                    border-radius: 15px; border: 1px solid rgba(34, 197, 94, 0.4);'>
            <span style='color: #22c55e; font-size: 1.8rem; font-weight: bold;'>😊</span><br>
            <span style='color: #22c55e; font-weight: 600;'>{len(POSITIVE_EXAMPLES)} Positive Examples</span><br>
            <span style='color: #94a3b8; font-size: 0.85rem;'>Next: #{(st.session_state.pos_idx % len(POSITIVE_EXAMPLES)) + 1}</span>
        </div>
        <div style='text-align: center; padding: 15px 30px; background: rgba(239, 68, 68, 0.2); 
                    border-radius: 15px; border: 1px solid rgba(239, 68, 68, 0.4);'>
            <span style='color: #ef4444; font-size: 1.8rem; font-weight: bold;'>😠</span><br>
            <span style='color: #ef4444; font-weight: 600;'>{len(NEGATIVE_EXAMPLES)} Negative Examples</span><br>
            <span style='color: #94a3b8; font-size: 0.85rem;'>Next: #{(st.session_state.neg_idx % len(NEGATIVE_EXAMPLES)) + 1}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Example Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Load Positive Example", use_container_width=True, key="pos_btn"):
            idx = st.session_state.pos_idx
            st.session_state.input_text = POSITIVE_EXAMPLES[idx]
            st.session_state.pos_idx = (idx + 1) % len(POSITIVE_EXAMPLES)
            st.session_state.last_result = None
            st.rerun()
    
    with col2:
        if st.button("😠 Load Negative Example", use_container_width=True, key="neg_btn"):
            idx = st.session_state.neg_idx
            st.session_state.input_text = NEGATIVE_EXAMPLES[idx]
            st.session_state.neg_idx = (idx + 1) % len(NEGATIVE_EXAMPLES)
            st.session_state.last_result = None
            st.rerun()
    
    with col3:
        if st.button("🧹 Clear All", use_container_width=True, key="clear_btn"):
            st.session_state.input_text = ""
            st.session_state.last_result = None
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Text Input - NO key parameter to allow session state to control the value
    user_input = st.text_area(
        "📝 Enter or paste your mobile product review here:",
        value=st.session_state.input_text,
        height=180,
        placeholder="Click an example button above OR type your review here...\n\nExample: The battery life is great and the camera takes amazing photos!"
    )
    
    # Update session state when user types (for manual input)
    st.session_state.input_text = user_input
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict Button - Centered and Prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True, key="predict_btn", type="primary")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Perform prediction
    if predict_clicked:
        if not st.session_state.input_text.strip():
            st.error("⚠️ Please enter a review or click an example button first!")
        else:
            with st.spinner("🔄 Analyzing sentiment..."):
                label, cleaned, confidence = predict_sentiment(st.session_state.input_text)
                st.session_state.last_result = {
                    'label': label,
                    'cleaned': cleaned,
                    'confidence': confidence,
                    'original': st.session_state.input_text
                }
                st.session_state.prediction_history.append({
                    'Review': st.session_state.input_text[:60] + '...' if len(st.session_state.input_text) > 60 else st.session_state.input_text,
                    'Sentiment': label.upper(),
                    'Confidence': f"{confidence:.1%}"
                })
    
    # Display Result
    if st.session_state.last_result:
        result = st.session_state.last_result
        
        if result['label'] == "positive":
            st.markdown(f"""
            <div class='success-prediction animate-fade'>
                <div style='font-size: 5rem; margin-bottom: 15px;'>😊</div>
                <h2 style='color: #22c55e; margin-bottom: 15px; font-size: 2rem;'>✅ POSITIVE SENTIMENT</h2>
                <p style='color: #cbd5e0; font-size: 1.15rem; max-width: 600px; margin: auto;'>
                    This review expresses <strong>satisfaction</strong> and <strong>positive emotions</strong> about the product.
                    The customer appears to be happy with their purchase!
                </p>
                <div style='margin-top: 25px; display: flex; justify-content: center; gap: 25px; flex-wrap: wrap;'>
                    <div style='padding: 20px 35px; background: rgba(34, 197, 94, 0.3); 
                                border-radius: 18px; border: 2px solid #22c55e;'>
                        <span style='font-size: 2rem; font-weight: 800; color: #22c55e;'>{result['confidence']:.1%}</span>
                        <br><span style='color: #cbd5e0; font-size: 0.95rem;'>Confidence Score</span>
                    </div>
                    <div style='padding: 20px 35px; background: rgba(34, 197, 94, 0.2); border-radius: 18px;'>
                        <span style='font-size: 2rem;'>👍</span>
                        <br><span style='color: #cbd5e0; font-size: 0.95rem;'>Recommended</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='error-prediction animate-fade'>
                <div style='font-size: 5rem; margin-bottom: 15px;'>😠</div>
                <h2 style='color: #ef4444; margin-bottom: 15px; font-size: 2rem;'>❌ NEGATIVE SENTIMENT</h2>
                <p style='color: #cbd5e0; font-size: 1.15rem; max-width: 600px; margin: auto;'>
                    This review expresses <strong>dissatisfaction</strong> and <strong>negative emotions</strong> about the product.
                    The customer appears to be unhappy with their experience.
                </p>
                <div style='margin-top: 25px; display: flex; justify-content: center; gap: 25px; flex-wrap: wrap;'>
                    <div style='padding: 20px 35px; background: rgba(239, 68, 68, 0.3); 
                                border-radius: 18px; border: 2px solid #ef4444;'>
                        <span style='font-size: 2rem; font-weight: 800; color: #ef4444;'>{result['confidence']:.1%}</span>
                        <br><span style='color: #cbd5e0; font-size: 0.95rem;'>Confidence Score</span>
                    </div>
                    <div style='padding: 20px 35px; background: rgba(239, 68, 68, 0.2); border-radius: 18px;'>
                        <span style='font-size: 2rem;'>👎</span>
                        <br><span style='color: #cbd5e0; font-size: 0.95rem;'>Not Recommended</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # NLP Preprocessing Details
        with st.expander("🔎 View NLP Preprocessing Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📝 Original Text")
                st.info(result['original'])
            with col2:
                st.markdown("#### 🧹 After Preprocessing")
                st.success(result['cleaned'])
            
            st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.15); padding: 20px; border-radius: 15px; margin-top: 15px;'>
                <strong style='color: #a78bfa; font-size: 1.1rem;'>🔧 Preprocessing Steps Applied:</strong>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;'>
                    <div style='color: #cbd5e0;'>✂️ Converted to lowercase</div>
                    <div style='color: #cbd5e0;'>🔤 Removed special characters</div>
                    <div style='color: #cbd5e0;'>🚫 Removed stopwords</div>
                    <div style='color: #cbd5e0;'>📝 Applied spell correction</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Business Strategies Section
        st.markdown("<br>", unsafe_allow_html=True)
        
        if result['label'] == "positive":
            st.markdown("""
            <div class='glass-card'>
                <h3 style='color: #22c55e;'>📈 Strategies to Maintain Positive Sentiment</h3>
                <div class='description-box'>
                    <strong>Business Insights:</strong> This positive review indicates customer satisfaction. 
                    Here are strategies to maintain and amplify this success:
                </div>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 20px;'>
                    <div style='background: rgba(34, 197, 94, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #22c55e;'>
                        <h4 style='color: #22c55e; margin-bottom: 10px;'>🌟 Leverage Testimonials</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Use positive reviews in marketing materials, social media, and product pages to build trust with new customers.</p>
                    </div>
                    <div style='background: rgba(34, 197, 94, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #22c55e;'>
                        <h4 style='color: #22c55e; margin-bottom: 10px;'>🎁 Reward Loyal Customers</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Implement loyalty programs, offer exclusive discounts, and create referral incentives for satisfied customers.</p>
                    </div>
                    <div style='background: rgba(34, 197, 94, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #22c55e;'>
                        <h4 style='color: #22c55e; margin-bottom: 10px;'>📊 Identify Success Factors</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Analyze what aspects customers love most (battery, camera, etc.) and emphasize these in future products.</p>
                    </div>
                    <div style='background: rgba(34, 197, 94, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #22c55e;'>
                        <h4 style='color: #22c55e; margin-bottom: 10px;'>💬 Engage & Thank</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Respond to positive reviews, thank customers personally, and encourage them to share their experience.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='glass-card'>
                <h3 style='color: #ef4444;'>🔧 Strategies to Improve & Address Negative Feedback</h3>
                <div class='description-box'>
                    <strong>Business Insights:</strong> This negative review highlights areas for improvement. 
                    Here are actionable strategies to turn criticism into opportunity:
                </div>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 20px;'>
                    <div style='background: rgba(239, 68, 68, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin-bottom: 10px;'>⚡ Immediate Response</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Respond quickly and empathetically. Acknowledge the issue, apologize sincerely, and offer a concrete solution or compensation.</p>
                    </div>
                    <div style='background: rgba(239, 68, 68, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin-bottom: 10px;'>🔍 Root Cause Analysis</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Identify the specific issues mentioned (battery, performance, quality) and prioritize fixes in your product development roadmap.</p>
                    </div>
                    <div style='background: rgba(239, 68, 68, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin-bottom: 10px;'>📞 Customer Recovery</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Reach out directly to unhappy customers. Offer replacements, refunds, or support to turn detractors into promoters.</p>
                    </div>
                    <div style='background: rgba(239, 68, 68, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin-bottom: 10px;'>📈 Quality Improvement</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Use negative feedback to improve QA processes, enhance product testing, and set higher quality standards.</p>
                    </div>
                    <div style='background: rgba(239, 68, 68, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin-bottom: 10px;'>📝 Update Documentation</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>If issues are related to user expectations, improve product descriptions, manuals, and feature explanations.</p>
                    </div>
                    <div style='background: rgba(239, 68, 68, 0.15); padding: 20px; border-radius: 15px; border-left: 4px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin-bottom: 10px;'>🛡️ Prevent Future Issues</h4>
                        <p style='color: #cbd5e0; font-size: 0.95rem;'>Implement early warning systems to detect product issues before they become widespread complaints.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
            <h3>📜 Prediction History</h3>
            <p style='color: #64748b;'>Last 5 predictions from this session</p>
        </div>
        """, unsafe_allow_html=True)
        
        history_df = pd.DataFrame(st.session_state.prediction_history[-5:][::-1])
        st.dataframe(history_df, use_container_width=True, hide_index=True)

# =================== FOOTER ===================

st.markdown("""
<div class='footer'>
    <p style='font-size: 1rem;'>Built with ❤️ using <strong>Streamlit</strong> | NLP Sentiment Analysis Dashboard</p>
    <p style='font-size: 0.9rem; color: #64748b;'>Model: TF-IDF + SMOTE + LinearSVC | Accuracy: 91%</p>
    <p style='font-size: 0.85rem; color: #64748b;'>© 2024 Shashank R | Data Science Portfolio Project</p>
</div>
""", unsafe_allow_html=True)
