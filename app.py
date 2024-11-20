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

# Register API blueprint
from api_routes import api_bp
app.register_blueprint(api_bp)

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
    
    # Import transliteration detector
    from transliteration_detector import detect_transliterated_abuse, get_transliteration_warning
    
    # Check for transliterated abuse FIRST
    is_transliterated_abuse, detected_words, translit_confidence = detect_transliterated_abuse(comment)
    
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
    
    is_bullying = any(score >= 0.5 for score in prediction)
    
    # If transliterated abuse detected, boost the scores
    if is_transliterated_abuse:
        # Boost toxic and insult scores based on transliteration confidence
        result['toxic'] = max(result['toxic'], translit_confidence * 0.9)
        result['insult'] = max(result['insult'], translit_confidence * 0.85)
        is_bullying = True  # Always flag as bullying if transliteration detected
    
    # Calculate severity score (0-100) - weighted average
    severity_score = int((
        result['toxic'] * 1.5 +  # toxic
        result['severe_toxic'] * 2.0 +  # severe_toxic (higher weight)
        result['obscene'] * 1.2 +  # obscene
        result['threat'] * 1.8 +  # threat (higher weight)
        result['insult'] * 1.3 +  # insult
        result['identity_hate'] * 1.6    # identity_hate (higher weight)
    ) / 10.4 * 100)  # Normalize to 0-100
    
    # Boost severity for transliterated abuse
    if is_transliterated_abuse:
        severity_score = max(severity_score, int(translit_confidence * 80))  # Minimum 80% severity
    
    # Clamp to 0-100 range
    severity_score = max(0, min(100, severity_score))
    
    response = {
        'result': result, 
        'is_bullying': bool(is_bullying),
        'severity_score': severity_score
    }
    
    # Add transliteration warning if detected
    if is_transliterated_abuse:
        response['transliteration_warning'] = get_transliteration_warning(detected_words)
        response['transliterated_abuse_detected'] = True
        response['detected_transliterated_words'] = detected_words
    
    return jsonify(response)

@app.route('/conclude', methods=['POST'])
def conclude():
    try:
        print("Conclude endpoint called")
        # Extract predictions from request
        predictions = request.json.get('result')
        is_bullying = request.json.get('is_bullying')

        # Build a summary of the predictions
        summary = f"The comment is{' not' if not is_bullying else ''} indicative of cyberbullying. "
        summary += "Here are the detailed probabilities for each type of toxicity: "
        for label, score in predictions.items():
            summary += f"{label.capitalize()}: {score:.2f}. "
        
        print(f"Summary: {summary}")
        
        # Generate conclusion using Google Gemini
        print("Initializing Gemini model...")
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        print("Sending request to Gemini API...")
        response = model.generate_content(
            f"Given the following summary of toxicity predictions for a comment, provide a concise conclusion (2-3 sentences max) about the nature of the comment. The prediction is based on a machine learning model. Here are the detailed probabilities: {summary}"
        )
        
        print("Received response from Gemini API")
        
        # Extract conclusion text
        if hasattr(response, 'text'):
            conclusion_text = response.text
        elif hasattr(response, '_result') and response._result.candidates:
            conclusion_text = response._result.candidates[0].content.parts[0].text
        else:
            raise ValueError("Unable to extract text from response")
        
        print(f"Conclusion: {conclusion_text}")
        return jsonify({'conclusion': conclusion_text})
        
    except Exception as e:
        print(f"Error in conclude endpoint: {str(e)}")
        # Return a fallback conclusion if Gemini fails
        fallback = f"The analysis indicates the comment is {'' if is_bullying else 'not '}likely cyberbullying based on the toxicity scores."
        return jsonify({'conclusion': fallback})

@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text to English for analysis"""
    try:
        from deep_translator import GoogleTranslator
        from langdetect import detect
    except ImportError:
        return jsonify({'error': 'Translation not available. Install: pip install deep-translator langdetect'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Detect language
        detected_lang = detect(text)
        
        if detected_lang != 'en':
            # Translate to English
            translated = GoogleTranslator(source=detected_lang, target='en').translate(text)
            return jsonify({
                'original': text,
                'translated': translated,
                'detected_language': detected_lang,
                'needs_translation': True
            })
        else:
            return jsonify({
                'original': text,
                'translated': text,
                'detected_language': 'en',
                'needs_translation': False
            })
    except Exception as e:
        print(f"Translation error: {str(e)}")
        # If translation fails, return original text
        return jsonify({
            'original': text,
            'translated': text,
            'detected_language': 'unknown',
            'needs_translation': False,
            'error': 'Translation failed, using original text'
        })

@app.route('/highlight_words', methods=['POST'])
def highlight_words():
    """Identify potentially toxic words in the text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Simple toxic words list (can be expanded)
        toxic_words = [
            'idiot', 'stupid', 'dumb', 'loser', 'hate', 'kill', 'die', 'shut up',
            'ugly', 'fat', 'worthless', 'pathetic', 'freak', 'trash', 'garbage',
            'moron', 'fool', 'disgusting', 'horrible'
        ]
        
        words = text.lower().split()
        highlighted = []
        
        for word in words:
            clean_word = re.sub(r'[^a-z]', '', word)
            if any(toxic in clean_word for toxic in toxic_words):
                highlighted.append(clean_word)
        
        return jsonify({
            'toxic_words': list(set(highlighted)),
            'count': len(highlighted)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
