# 🔒 Predicting Cyberbullying on Social Media - Flask Web App

![Cyberbullying Detection](https://github.com/chandu080201/Predicting_Cyberbullying_Infosys_Internship_Oct24/blob/main/WhatsApp%20Image%202025-03-22%20at%2022.15.30.gif)

### 🚀 End-to-End Flask-Based AI Application for Detecting Cyberbullying on Social Media Platforms

**This project leverages Machine Learning and NLP techniques to identify cyberbullying in social media text.** Built with Flask, the web application allows users to input social media text and receive real-time analysis, categorizing messages as non-toxic or cyberbullying-related.

---

## 📌 Project Overview

Cyberbullying is a growing concern in today's digital world. This Flask web application enables real-time cyberbullying detection using **Machine Learning models**, including **Random Forest (RF), Logistic Regression (LR), Support Vector Machines (SVM), Gradient Boosting, and Naive Bayes**.

### 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-%E2%9C%94-black?style=for-the-badge&logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-%E2%9C%94-purple?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-%E2%9C%94-red?style=for-the-badge)
![Natural Language Processing](https://img.shields.io/badge/NLP-%E2%9C%94-green?style=for-the-badge)

---

## 💡 Features

✅ **Real-Time Cyberbullying Detection** - Input a message and get immediate analysis.
✅ **Advanced Machine Learning Models** - Uses RF, LR, SVM, Gradient Boosting, and Naive Bayes for high accuracy.
✅ **Interactive Flask Web Interface** - Easy-to-use UI for text classification.
✅ **Data Preprocessing & NLP** - Removes stopwords, tokenizes, and applies TF-IDF.
✅ **Dashboard & Visualization** - Displays trends in cyberbullying detection.

---

## 🌐 Live Demo (Example Output)

**Input:** "You're such a loser, no one likes you."

**Prediction:** 🚨 **Cyberbullying Detected! (Toxic Behavior)**

**Input:** "Hey, I hope you have a great day!"

**Prediction:** ✅ **Safe Message (Non-Toxic)**

---


## 🎨 System Architecture

**Architecture Overview:**
1. **Data Collection & Preprocessing**: Collect social media text, clean data, and apply NLP techniques (tokenization, stopword removal, TF-IDF, etc.).
2. **Model Training & Selection**: Train multiple ML models (Random Forest, SVM, Logistic Regression, etc.) and choose the best-performing one.
3. **Web Application Development**: Integrate the trained model into a Flask-based web app for real-time analysis.
4. **User Interaction**: Accept text input from users and display classification results.
5. **Deployment & Monitoring**: Deploy on a cloud server and monitor performance over time.

---

## 🔄 Installation & Setup

### ➕ 1. Clone the Repository
```bash
git clone https://github.com/chandu080201/Predicting_Cyberbullying_Infosys_Internship_Oct24.git
cd Predicting_Cyberbullying_Infosys_Internship_Oct24
```

### ➕ 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### ➕ 3. Run the Flask App
```bash
python app.py
```

📍 Open **`localhost:5000`** in your browser.

---

## 🎮 Usage Guide

1️⃣ **Enter Social Media Text** - Type a comment, tweet, or message.
2️⃣ **Click "Analyze"** - The model processes the text.
3️⃣ **View Results** - The app predicts whether it's cyberbullying.

---

## 🌌 Future Enhancements

🚀 **Real-Time Twitter Monitoring** - Monitor live tweets for cyberbullying.
🚀 **Multilingual Cyberbullying Detection** - Expand to multiple languages.
🚀 **Improved Model Explainability** - Use SHAP/LIME for insights.

---

## 💪 Contributing

👉 Fork the repo  
👉 Create a new branch (`feature/new-feature`)  
👉 Submit a PR 🎉  
