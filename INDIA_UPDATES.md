# India - Specific Resources Update

## Changes Made

### 1. Resources Section
**Updated** [`templates/index.html`](file:///c:/Users/CHANDU/Desktop/CYBER-PRED/Predicting_Cyberbullying_Infosys_Internship_Oct24/templates/index.html)

Replaced US-based resources with India-specific ones:

- **📞 Childline India** - Call 1098 (24/7 helpline)
- **🛡️ Cyber Crime Portal** - Report at cybercrime.gov.in
- **🆘 NCPCR** - National Commission for Protection of Child Rights
- **💬 iCall Mental Health** - Call 9152987821 (Mon-Sat, 10am-8pm)
- **🏫 NCERT Guidance** - Cybersafety resources for students
- **👨‍👩‍👧 Parents Guide** - UNICEF India safer internet guide

**Emergency Contact Footer:**
- Women Helpline: 181
- Child Helpline: 1098
- Cyber Crime: 155260

### 2. Multi-language Translation Fixed

**Updated** [`app.py`](file:///c:/Users/CHANDU/Desktop/CYBER-PRED/Predicting_Cyberbullying_Infosys_Internship_Oct24/app.py)
- Switched from `googletrans` to `deep-translator` (more reliable)
- Added `langdetect` for language detection
- Improved error handling

**Updated** [`requirements.txt`](file:///c:/Users/CHANDU/Desktop/CYBER-PRED/Predicting_Cyberbullying_Infosys_Internship_Oct24/requirements.txt)
- Added `deep-translator`
- Added `langdetect`

## How to Use Translation

### Via API:
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "చంపేస్తా నక్కోద్దా"}'
```

Response:
```json
{
  "original": "చంపేస్తా నక్కోద్దా",
  "translated": "I will kill you",
  "detected_language": "te",
  "needs_translation": true
}
```

## Installation

Install the new dependencies:
```bash
pip install deep-translator langdetect
```

Or:
```bash
pip install -r requirements.txt
```

## Testing Multi-language

Test with various Indian languages:
- Telugu: "చంపేస్తా నక్కోద్దా"
- Hindi: "तुम बेवकूफ हो"
- Tamil: "நீ முட்டாள்"
- Kannada: "ನೀನು ಮೂರ್ಖ"

The system will automatically:
1. Detect the language 
2. Translate to English
3. Analyze for cyberbullying
4. Return results

## Next Steps

Test the translation by:
1. Restart Flask app (`python app.py`)
2. Enter text in an Indian language
3. The backend `/translate` endpoint is ready for integration
4. (Optional) Add a translate button in the UI to call this endpoint before prediction
