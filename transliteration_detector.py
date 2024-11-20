"""
Transliteration Detection for Cyberbullying Prevention
Handles cases like "neevu moorkudavu" that evade language detection
"""

# Common abusive words in transliteration across Indian languages
TRANSLITERATED_ABUSIVE_WORDS = {
    # Telugu transliterations
    'moorkudu': 'fool/idiot',
    'moorkudavu': 'you fool',
    'buddi': 'stupid',
    'lanjakoduku': 'offensive',
    'gorrelu': 'sheep/dumb',
    
    # Hindi transliterations
    'bewakoof': 'fool',
    'pagal': 'crazy',
    'kamina': 'vile',
    'chutiya': 'offensive',
    'gandu': 'offensive',
    'madarchod': 'offensive',
    'bhosdi': 'offensive',
    
    # Tamil transliterations
    'muttaal': 'fool',
    'loosu': 'stupid',
    'paithiyam': 'crazy',
    'puluthi': 'dirty',
    
    # Kannada transliterations
    'moorka': 'fool',
    'huccha': 'crazy',
    'bekku': 'offensive',
    
    # Bengali transliterations
    'pagol': 'crazy',
    'gadha': 'donkey/fool',
    'choda': 'offensive',
    
    # Malayalam transliterations
    'patti': 'dog',
    'thenga': 'coconut/fool',
    
    # General abusive patterns (leetspeak/intentional misspellings)
    'f00l': 'fool',
    'st0pid': 'stupid',
    'idi0t': 'idiot',
    'dum8': 'dumb',
}

# Phonetic variations (common misspellings to evade detection)
PHONETIC_VARIATIONS = {
    'bewakuf': 'bewakoof',
    'morkudu': 'moorkudu',
    'kamena': 'kamina',
    'pagle': 'pagal',
}

def detect_transliterated_abuse(text):
    """
    Detect abusive transliterated words in text
    Returns: (is_abusive, detected_words, confidence_score)
    """
    text_lower = text.lower()
    detected = []
    
    # Check for direct matches
    for word in TRANSLITERATED_ABUSIVE_WORDS:
        if word in text_lower:
            detected.append({
                'word': word,
                'meaning': TRANSLITERATED_ABUSIVE_WORDS[word],
                'position': text_lower.find(word)
            })
    
    # Check for phonetic variations
    for variant, original in PHONETIC_VARIATIONS.items():
        if variant in text_lower and original in TRANSLITERATED_ABUSIVE_WORDS:
            detected.append({
                'word': variant,
                'meaning': TRANSLITERATED_ABUSIVE_WORDS[original],
                'position': text_lower.find(variant),
                'note': f'Phonetic variant of "{original}"'
            })
    
    # Calculate confidence based on number of matches
    is_abusive = len(detected) > 0
    confidence = min(len(detected) * 0.4, 1.0)  # 0.4 per word, max 1.0
    
    return is_abusive, detected, confidence

def get_transliteration_warning(detected_words):
    """
    Generate a warning message for detected transliterated abuse
    """
    if not detected_words:
        return None
    
    words_list = [f"'{w['word']}'" for w in detected_words]
    return {
        'warning': 'Potentially abusive transliterated content detected',
        'detected_words': words_list,
        'recommendation': 'This text may contain offensive content in transliterated form',
        'severity': 'HIGH' if len(detected_words) >= 2 else 'MEDIUM'
    }
