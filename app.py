from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('nmodel.pkl')
vectorizer = joblib.load('nvectorizer.pkl')

# Initialize NLTK components
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def clean_text(text):
    """Cleans the input text."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\w*\d\w*', '', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  
    text = re.sub(r'\n', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text

def preprocess_input(text):
    """Preprocess the input text for prediction."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    processed_text = ' '.join(tokens)
    vectorized_text = vectorizer.transform([processed_text])
    return vectorized_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    
    # Preprocess and vectorize the input comment
    vectorized_text = preprocess_input(comment)
    prediction = model.predict_proba(vectorized_text)[0]
    
    result = {
        'toxic': prediction[0],
        'severe_toxic': prediction[1],
        'obscene': prediction[2],
        'threat': prediction[3],
        'insult': prediction[4],
        'identity_hate': prediction[5]
    }
    
    is_bullying = prediction[6]>=0.5

    return jsonify({'result': result, 'is_bullying': bool(is_bullying)})

@app.route('/conclude', methods=['POST'])
def conclude():
    print("called")
    # Extract predictions from request
    predictions = request.json.get('result')
    is_bullying = request.json.get('is_bullying')

    # Build a summary of the predictions
    summary = f"The comment is{' not' if not is_bullying else ''} indicative of cyberbullying. "
    summary += "Here are the detailed probabilities for each type of toxicity: "
    for label, score in predictions.items():
        summary += f"{label.capitalize()}: {score:.2f}. "
    # Generate conclusion using Google Gemini-Pro
    print(summary)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"Given the following summary of toxicity predictions for a comment, provide a concise conclusion about the nature of the comment. The prediction is based on a machine learning model, and you should communicate the certainty of the prediction. Here are the detailed probabilities for each type of toxicity, also do not include the technical aspects like what models predict: {summary}"
    )
    print(response)
    try:
        candidates = response._result.candidates
        conclusion_text = candidates[0].content.parts[0].text
    except AttributeError as e:
        return jsonify({'error': 'Failed to extract conclusion text from the response: ' + str(e)}), 500
    except IndexError as e:
        return jsonify({'error': 'Index out of range when extracting conclusion text: ' + str(e)}), 500
    print(conclusion_text)
    return jsonify({'conclusion': conclusion_text})

if __name__ == '__main__':
    app.run(debug=True)
