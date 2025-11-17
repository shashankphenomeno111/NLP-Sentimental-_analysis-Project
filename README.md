<h1 align="center" style="font-size: 60px; font-weight: 900;">
  📱 Sentiment Analysis Using SVM
</h1>

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/d7714ff8-81ea-4b39-87ad-edca3a9c0d2f" />
<!-- BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-SVM-purple?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF-purple?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-purple?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Accuracy-91%25-brightgreen?style=for-the-badge&logo=target" />
</p>

---

<p align="center">
  <a href=👉(https://nlp-sentimental-analysis-project.streamlit.app/) target="_blank">
    <img src="https://img.shields.io/badge/🔴 LIVE DEMO-Streamlit%20App-purple?style=for-the-badge&logo=streamlit" />
  </a>
</p>

## 🎯 Objective

The objective of this project is to build an efficient and accurate sentiment analysis system for mobile product reviews using classical Machine Learning techniques. The goal is to automatically classify user reviews into **Positive** or **Negative** sentiments by applying:

- Text preprocessing and spell correction  
- TF-IDF vectorization for feature extraction  
- SMOTE for class balancing  
- Linear SVM for high-accuracy classification  

The final system is deployed as an interactive **Streamlit Web Application**, enabling users to input any mobile review and instantly receive a sentiment prediction with a clean, user-friendly interface.


## 🚀 Live Demo

🔗 **Try the App Here:**  
👉(https://nlp-sentimental-analysis-project.streamlit.app/)


# 🌟 Overview

This project is a **Sentiment Analysis Web Application** built using  
**TF-IDF + SMOTE + LinearSVC** and deployed using **Streamlit Cloud**.

It predicts whether a mobile review is:

- ✔ **Positive**
- ❌ **Negative**

🔥 Also includes:
- 10 rotating sample reviews (positive & negative)
- Auto spell-correction for user input
- Cleaned text viewer (for NLP explanations)
- Modern UI with sidebar summary

---

# 🎨 Features

<p align="center">
<table>
  <tr>
    <td><b>🔥 High Accuracy (91%)</b></td>
    <td><b>💡 Spell Correction</b></td>
    <td><b>⚡ Fast Predictions</b></td>
  </tr>
  <tr>
    <td><b>📊 TF-IDF + SMOTE</b></td>
    <td><b>🤖 LinearSVC</b></td>
    <td><b>🌐 Streamlit Deployment</b></td>
  </tr>
</table>
</p>

---

# 🧠 Tech Stack

✔ **Python**  
✔ **Scikit-Learn**  
✔ **NLTK**  
✔ **SMOTE (imblearn)**  
✔ **Streamlit**  
✔ **Joblib**  

---

# 🔮 Workflow

<p align="center">
  <img width="600" height="297" alt="image" src="https://github.com/user-attachments/assets/e5b88e4b-a372-4d77-b335-b0cf41be641f" />

</p>

Dataset
↓
Rating → Positive / Negative
↓
Text Cleaning + Spell-Correction
↓
TF-IDF Vectorization (1–2 grams)
↓
SMOTE Oversampling
↓
Linear SVM Training
↓
Model Saving (joblib)
↓
Streamlit Web App
↓
User Input → Sentiment Prediction

yaml
Copy code

---

# 📸 App UI Preview

<img width="1895" height="1011" alt="image" src="https://github.com/user-attachments/assets/b1bc5496-cc31-4c26-9533-5884767aad0b" />


---

# 📊 Model Performance

| Metric | Score |
|-------|-------|
| **Accuracy** | 91% |
| Precision | 0.91 |
| Recall | 0.90 |
| F1 Score | 0.90 |

---

# 🧩 Project Structure

📁 sentiment-analysis-svm
│── app.py
│── sentiment_svm_model.pkl
│── tfidf_vectorizer.pkl
│── requirements.txt
│── README.md

yaml
Copy code

---

# 🚀 Installation

pip install -r requirements.txt
streamlit run app.py

yaml
Copy code

---

# 🌐 Deployment (Streamlit Cloud)

1. Push to GitHub  
2. Go to **share.streamlit.io**  
3. Select repository → choose **app.py**  
4. Add requirements.txt  
5. Deploy 🎉  

---

# 🧑‍💻 Author

**Shashank R**  
💼 Data Science Enthusiast  
<p align="center">
  <a href="https://github.com/shashankphenomeno111" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Shashank%20R-181717?style=for-the-badge&logo=github" />
  </a>
  <a href="https://www.linkedin.com/in/shashankdatascientist/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Shashank%20R-0A66C2?style=for-the-badge&logo=linkedin" />
  </a>
</p>
 

---

# ⭐ If you like this project, give it a star! ⭐

<p align="center">
  ⭐⭐⭐⭐⭐
</p>
