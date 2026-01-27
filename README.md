<h1 align="center">
  <img src="https://github.com/user-attachments/assets/d7714ff8-81ea-4b39-87ad-edca3a9c0d2f" alt="NLP Banner" width="100%"/>
  <br/>
  📱 NLP Sentiment Analysis Dashboard
  <br/>
  <sub>Mobile Product Review Classification using Machine Learning</sub>
</h1>

<p align="center">
  <a href="https://nlp-sentimental-analysis-project.streamlit.app/">
    <img src="https://img.shields.io/badge/🔴 LIVE DEMO-Streamlit Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine_Learning-SVM-9146FF?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="ML"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF-00D4AA?style=for-the-badge&logo=spacy&logoColor=white" alt="NLP"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Accuracy-91%25-success?style=for-the-badge&logo=target&logoColor=white" alt="Accuracy"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=flat-square" alt="Contributions"/>
</p>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🔬 Methodology](#-methodology)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🏗️ Technical Architecture](#️-technical-architecture)
- [⚙️ Model Details](#️-model-details)
- [📈 Results & Performance](#-results--performance)
- [🎨 Dashboard Features](#-dashboard-features)
- [🚀 Installation & Setup](#-installation--setup)
- [☁️ Deployment](#️-deployment)
- [📁 Project Structure](#-project-structure)
- [🔮 Future Improvements](#-future-improvements)
- [👨‍💻 Author](#-author)

---

## 🎯 Project Overview

### Problem Statement

In the e-commerce industry, understanding customer sentiment is crucial for product improvement and business growth. Manual analysis of thousands of reviews is time-consuming and inefficient. This project addresses this challenge by automating sentiment classification using **Natural Language Processing (NLP)** and **Machine Learning**.

### Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Build an accurate sentiment classification model for mobile reviews | ✅ Achieved (91% accuracy) |
| 2 | Implement comprehensive text preprocessing pipeline | ✅ Completed |
| 3 | Handle class imbalance using oversampling techniques | ✅ SMOTE implemented |
| 4 | Deploy as an interactive web application | ✅ Streamlit Cloud |
| 5 | Create informative visualizations for EDA | ✅ Dashboard ready |

### What This Project Does

```
📝 Customer Review  ──→  🧹 NLP Preprocessing  ──→  📊 TF-IDF Features  ──→  🤖 LinearSVC  ──→  ✅ Sentiment
                                                                                              (Positive/Negative)
```

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔥 Core Features
- **91% Accuracy** on test data
- **Real-time predictions** with confidence scores
- **Spell correction** for noisy input
- **Interactive dashboard** with 4 tabs
- **Comprehensive EDA** visualizations

</td>
<td width="50%">

### 🛠️ Technical Features
- **TF-IDF Vectorization** (uni + bi-grams)
- **SMOTE** class balancing
- **LinearSVC** classifier
- **Streamlit Cloud** deployment
- **Session-based** prediction history

</td>
</tr>
</table>

---

## 🔬 Methodology

### 1️⃣ Data Collection

- **Source**: Amazon mobile product reviews
- **Size**: 1,440 reviews
- **Features**: Title, Rating (1-5), Review Text

### 2️⃣ Data Preprocessing Pipeline

```
Raw Text
    │
    ▼
┌─────────────────────────────────┐
│  1. Lowercase Conversion        │
│  2. Remove Special Characters   │
│  3. Tokenization                │
│  4. Stopword Removal            │
│  5. Spell Correction            │
└─────────────────────────────────┘
    │
    ▼
Clean Text
```

**Key Preprocessing Steps:**

| Step | Description | Example |
|------|-------------|---------|
| Lowercase | Convert to lowercase | "GREAT Phone!" → "great phone!" |
| Clean | Remove non-alphabetic chars | "phone123!" → "phone" |
| Tokenize | Split into words | "great phone" → ["great", "phone"] |
| Stopwords | Remove common words | ["the", "is", "a"] removed |
| Spell Check | Fix typos | "battrey" → "battery" |

### 3️⃣ Feature Engineering

**TF-IDF (Term Frequency-Inverse Document Frequency)**

$$TF\text{-}IDF(t,d) = TF(t,d) \times \log\left(\frac{N}{DF(t)}\right)$$

Where:
- `TF(t,d)` = Term frequency of term t in document d
- `N` = Total number of documents
- `DF(t)` = Number of documents containing term t

**Configuration:**
- N-grams: (1, 2) - Captures both single words and word pairs
- Max Features: 5,000 most important terms
- Sublinear TF: Enabled for better scaling

### 4️⃣ Class Balancing with SMOTE

**Why SMOTE?**
- Original data had class imbalance (more positive reviews)
- SMOTE generates synthetic samples for minority class
- Improves model's ability to detect negative sentiment

```
Before SMOTE          After SMOTE
┌─────────────┐       ┌─────────────┐
│ Positive: 800│      │ Positive: 800│
│ Negative: 400│  ──→ │ Negative: 800│
└─────────────┘       └─────────────┘
```

### 5️⃣ Model Training

**LinearSVC (Linear Support Vector Classifier)**
- Finds optimal hyperplane to separate classes
- Efficient for high-dimensional text data
- Fast training and inference

---

## 📊 Exploratory Data Analysis

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,440 |
| Features | 3 (title, rating, reviews) |
| Target Classes | 2 (Positive, Negative) |
| Avg. Review Length | ~250 characters |
| Rating Range | 1-5 stars |

### Rating Distribution

```
⭐⭐⭐⭐⭐ (5) ████████████████████████ 450
⭐⭐⭐⭐   (4) █████████████████ 320
⭐⭐⭐     (3) ████████ 180
⭐⭐       (2) ███████ 150
⭐         (1) ██████████████ 340
```

### Sentiment Split

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| 😊 Positive (4-5 stars) | 770 | 53.5% |
| 😠 Negative (1-3 stars) | 670 | 46.5% |

### Key EDA Insights

1. **Rating Polarization**: Reviews are mostly either very positive (5 stars) or very negative (1 star)
2. **Text Length Correlation**: Negative reviews tend to be longer (more complaints to describe)
3. **Common Positive Words**: "battery", "camera", "excellent", "display", "fast"
4. **Common Negative Words**: "problem", "issue", "bad", "poor", "waste"

---

## 🏗️ Technical Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                   STREAMLIT CLOUD                        │
                         │  ┌─────────────────────────────────────────────────────┐ │
                         │  │              STREAMLIT WEB APP (app.py)             │ │
                         │  │  ┌───────────┬───────────┬───────────┬───────────┐  │ │
                         │  │  │   HOME    │    EDA    │   MODEL   │ PREDICTOR │  │ │
                         │  │  │   TAB     │    TAB    │    TAB    │    TAB    │  │ │
                         │  │  └───────────┴───────────┴───────────┴───────────┘  │ │
                         │  └─────────────────────────┬───────────────────────────┘ │
                         │                            │                             │
                         │  ┌─────────────────────────▼───────────────────────────┐ │
                         │  │                   ML PIPELINE                        │ │
                         │  │  ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │ │
                         │  │  │ TF-IDF   │ → │  SMOTE   │ → │    LinearSVC     │  │ │
                         │  │  │Vectorizer│   │ Balancer │   │   Classifier     │  │ │
                         │  │  └──────────┘   └──────────┘   └──────────────────┘  │ │
                         │  └─────────────────────────────────────────────────────┘ │
                         │                                                          │
                         │  ┌─────────────────────────────────────────────────────┐ │
                         │  │               SAVED ARTIFACTS (.pkl)                 │ │
                         │  │  • sentiment_svm_model.pkl                          │ │
                         │  │  • tfidf_vectorizer.pkl                             │ │
                         │  └─────────────────────────────────────────────────────┘ │
                         └─────────────────────────────────────────────────────────┘
```

### Component Descriptions

| Component | Purpose | Technology |
|-----------|---------|------------|
| Web Interface | User interaction & visualization | Streamlit |
| Text Preprocessor | Clean & normalize text | NLTK, Regex |
| Feature Extractor | Convert text to numbers | TF-IDF Vectorizer |
| Class Balancer | Handle imbalanced data | SMOTE |
| Classifier | Predict sentiment | LinearSVC |
| Model Storage | Persist trained models | Joblib (pickle) |

---

## ⚙️ Model Details

### TF-IDF Vectorizer

```python
TfidfVectorizer(
    ngram_range=(1, 2),      # Unigrams and Bigrams
    max_features=5000,       # Top 5000 features
    sublinear_tf=True,       # Use log(1 + tf)
    min_df=2,                # Minimum document frequency
    max_df=0.95              # Maximum document frequency
)
```

**Why these parameters?**
- `ngram_range=(1,2)`: Captures both single words ("excellent") and phrases ("battery life")
- `max_features=5000`: Balance between information and dimensionality
- `sublinear_tf=True`: Prevents common words from dominating

### SMOTE Configuration

```python
SMOTE(
    sampling_strategy='auto',  # Balance classes
    k_neighbors=5,             # Neighbors for synthesis
    random_state=42            # Reproducibility
)
```

### LinearSVC Classifier

```python
LinearSVC(
    C=1.0,                     # Regularization parameter
    class_weight='balanced',   # Handle imbalance
    max_iter=1000,            # Maximum iterations
    random_state=42           # Reproducibility
)
```

---

## 📈 Results & Performance

### Overall Metrics

<table>
<tr>
<td align="center">
<h2>91%</h2>
<b>Accuracy</b>
</td>
<td align="center">
<h2>0.91</h2>
<b>Precision</b>
</td>
<td align="center">
<h2>0.90</h2>
<b>Recall</b>
</td>
<td align="center">
<h2>0.90</h2>
<b>F1-Score</b>
</td>
</tr>
</table>

### Confusion Matrix

```
                 Predicted
              Neg      Pos
           ┌────────┬────────┐
    Neg    │   85   │    9   │   True Negatives / False Positives
Actual     ├────────┼────────┤
    Pos    │   10   │   96   │   False Negatives / True Positives
           └────────┴────────┘
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.89 | 0.90 | 0.90 | 94 |
| Positive | 0.91 | 0.91 | 0.91 | 106 |
| **Weighted Avg** | **0.91** | **0.90** | **0.90** | **200** |

### Performance Analysis

✅ **Strengths:**
- High accuracy across both classes
- Good balance between precision and recall
- Fast inference time (~10ms per prediction)

⚠️ **Limitations:**
- Binary classification only (no neutral class)
- Domain-specific (mobile reviews)
- May struggle with sarcasm/irony

---

## 🎨 Dashboard Features

### Tab 1: 🏠 Home

<table>
<tr>
<td width="60%">

**What's Included:**
- Project overview and objectives
- Quick statistics cards
- Technology stack badges
- Usage instructions
- ML pipeline summary

</td>
<td width="40%">

**Metrics Displayed:**
- Total reviews: 1,440
- Positive %: 53.5%
- Negative %: 46.5%
- Model accuracy: 91%

</td>
</tr>
</table>

### Tab 2: 📊 EDA Analysis

**Visualizations with Descriptions:**

| Section | Description |
|---------|-------------|
| Dataset Overview | Sample size, features, data types |
| Rating Distribution | Bar chart with insights |
| Sentiment Split | Pie chart with percentages |
| Text Length Analysis | Histogram with statistics |
| Word Clouds | Positive vs Negative word clouds |
| Top Words | Horizontal bar charts by sentiment |

### Tab 3: 🤖 Model Information

**Sections:**
- Model architecture explanation
- Pipeline flowchart visualization
- Performance metrics with descriptions
- Confusion matrix heatmap
- Classification report table

### Tab 4: 🔍 Sentiment Prediction

**Features:**
- Text input area
- Example buttons (Positive/Negative)
- Clear button
- Real-time prediction
- Confidence score display
- Preprocessed text viewer
- Prediction history (session-based)

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone Repository

```bash
git clone https://github.com/shashankphenomeno111/NLP-Sentimental-_analysis-Project.git
cd NLP-Sentimental-_analysis-Project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
```

### Step 5: Run Application

```bash
streamlit run app.py
```

### Step 6: Access Dashboard

Open browser and navigate to: `http://localhost:8501`

---

## ☁️ Deployment

### Live Application

🔗 **[https://nlp-sentimental-analysis-project.streamlit.app/](https://nlp-sentimental-analysis-project.streamlit.app/)**

### Deploy on Streamlit Cloud (Step-by-Step)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add sentiment analysis dashboard"
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Create New App**
   - Click "New app"
   - Select your repository
   - Branch: `main`
   - Main file path: `app.py`

4. **Add Secrets (if needed)**
   - Click "Advanced settings"
   - Add any API keys or secrets

5. **Deploy! 🎉**
   - Click "Deploy"
   - Wait for build to complete
   - Share your app URL

### Requirements for Deployment

Ensure `requirements.txt` contains:
```
streamlit
pandas
numpy
scikit-learn
nltk
imbalanced-learn
pyspellchecker
joblib
plotly
wordcloud
matplotlib
```

---

## 📁 Project Structure

```
NLP-Sentimental-_analysis-Project/
│
├── 📊 app.py                          # Streamlit dashboard application
│                                       # - 4 interactive tabs
│                                       # - EDA visualizations
│                                       # - Real-time predictions
│
├── 📓 Sentimental_Analysis (1).ipynb  # Jupyter notebook with full analysis
│                                       # - Data exploration
│                                       # - Model training
│                                       # - Evaluation metrics
│
├── 🤖 sentiment_svm_model.pkl         # Trained LinearSVC model
│                                       # - Serialized with joblib
│                                       # - Ready for inference
│
├── 📝 tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
│                                       # - Same configuration as training
│                                       # - Required for text transformation
│
├── 📊 dataset -P582 (1).csv           # Mobile review dataset
│                                       # - 1,440 reviews
│                                       # - Columns: title, rating, reviews
│
├── 📋 requirements.txt                # Python dependencies
│                                       # - All required packages
│                                       # - Version specifications
│
└── 📖 README.md                       # Project documentation (this file)
                                        # - Comprehensive guide
                                        # - Setup instructions
```

---

## 🔮 Future Improvements

| Priority | Enhancement | Description |
|----------|-------------|-------------|
| 🔴 High | Multi-class Sentiment | Add "Neutral" category for 3-class classification |
| 🔴 High | Deep Learning Model | Implement BERT/RoBERTa for improved accuracy |
| 🟡 Medium | Multi-language Support | Extend to Hindi, Spanish, etc. |
| 🟡 Medium | Aspect-based Analysis | Identify sentiment for specific features (camera, battery) |
| 🟢 Low | Batch Processing | Upload CSV for bulk predictions |
| 🟢 Low | API Endpoint | REST API for integration |

### Roadmap

```
Q1 2024: ✅ Binary Classification Model (Current)
Q2 2024: 🔄 Add Neutral Class & Improve Accuracy
Q3 2024: 📝 Implement Deep Learning (BERT)
Q4 2024: 🌍 Multi-language & API Support
```

---

## 🛠️ Tech Stack Summary

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Sklearn"/>
  <img src="https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="NLTK"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
</p>

---

## 👨‍💻 Author

<p align="center">
  <img src="https://github.com/shashankphenomeno111.png" width="150" style="border-radius: 50%;" alt="Author"/>
</p>

<h3 align="center">Shashank R</h3>
<p align="center">
  <b>Data Science Enthusiast | ML Engineer | NLP Practitioner</b>
</p>

<p align="center">
  <a href="https://github.com/shashankphenomeno111" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
  <a href="https://www.linkedin.com/in/shashankdatascientist/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
</p>

---

## 🙏 Acknowledgments

- **Dataset**: Amazon Mobile Reviews
- **Libraries**: Scikit-learn, NLTK, Streamlit, Plotly
- **Deployment**: Streamlit Cloud
- **Inspiration**: E-commerce sentiment analysis use cases

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>⭐ If you found this project helpful, please give it a star! ⭐</b>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/shashankphenomeno111/NLP-Sentimental-_analysis-Project?style=social" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/shashankphenomeno111/NLP-Sentimental-_analysis-Project?style=social" alt="Forks"/>
  <img src="https://img.shields.io/github/watchers/shashankphenomeno111/NLP-Sentimental-_analysis-Project?style=social" alt="Watchers"/>
</p>

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/shashankphenomeno111">Shashank R</a>
</p>
