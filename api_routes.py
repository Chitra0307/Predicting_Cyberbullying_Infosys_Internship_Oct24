"""
API Routes for Cyberbullying Detection
REST API endpoint with authentication and additional features
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import os

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Simple API key authentication
API_KEY = os.getenv('API_KEY', 'demo_api_key_12345')  # Change this in production

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('X-API-Key')
        if not auth_header or auth_header != API_KEY:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@api_bp.route('/predict', methods=['POST'])
@require_api_key
def api_predict():
    """
    API endpoint for cyberbullying prediction
    
    Request JSON:
    {
        "comment": "text to analyze"
    }
    
    Response JSON:
    {
        "result": {
            "toxic": float,
            "severe_toxic": float,
            "obscene": float,
            "threat": float,
            "insult": float,
            "identity_hate": float
        },
        "is_bullying": bool,
        "severity_score": int (0-100)
    }
    """
    from app import model, vectorizer, preprocess_input
    
    data = request.get_json()
    if not data or 'comment' not in data:
        return jsonify({'error': 'Missing comment in request body'}), 400
    
    comment = data['comment']
    
    try:
        # Preprocess and vectorize
        vectorized_text = preprocess_input(comment)
        prediction = model.predict_proba(vectorized_text)[0]
        
        result = {
            'toxic': float(prediction[0]),
            'severe_toxic': float(prediction[1]),
            'obscene': float(prediction[2]),
            'threat': float(prediction[3]),
            'insult': float(prediction[4]),
            'identity_hate': float(prediction[5])
        }
        
        is_bullying = any(score >= 0.5 for score in prediction)
        
        # Calculate severity score
        severity_score = int((
            prediction[0] * 1.5 +
            prediction[1] * 2.0 +
            prediction[2] * 1.2 +
            prediction[3] * 1.8 +
            prediction[4] * 1.3 +
            prediction[5] * 1.6
        ) / 10.4 * 100)
        
        severity_score = max(0, min(100, severity_score))
        
        return jsonify({
            'result': result,
            'is_bullying': bool(is_bullying),
            'severity_score': severity_score
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
