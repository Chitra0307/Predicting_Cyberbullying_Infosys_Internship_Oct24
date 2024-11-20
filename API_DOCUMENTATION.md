# Cyberbullying Detection API Documentation

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication

All API requests require an API key passed in the `X-API-Key` header.

```bash
X-API-Key: your_api_key_here
```

Default API key for demo: `demo_api_key_12345` (Change in production!)

---

## Endpoints

### 1. Predict Cyberbullying

**Endpoint:** `POST /api/v1/predict`

**Description:** Analyzes text for cyberbullying and returns toxicity scores.

**Request:**
```json
{
  "comment": "Your text to analyze here"
}
```

**Response:**
```json
{
  "result": {
    "toxic": 0.87,
    "severe_toxic": 0.19,
    "obscene": 0.45,
    "threat": 0.02,
    "insult": 0.85,
    "identity_hate": 0.12
  },
  "is_bullying": true,
  "severity_score": 68
}
```

**Example using curl:**
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo_api_key_12345" \
  -d '{"comment": "You stupid idiot"}'
```

**Example using Python:**
```python
import requests

url = "http://localhost:5000/api/v1/predict"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "demo_api_key_12345"
}
data = {
    "comment": "Your text here"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

**Example using JavaScript:**
```javascript
fetch('http://localhost:5000/api/v1/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'demo_api_key_12345'
  },
  body: JSON.stringify({
    comment: 'Your text here'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### 2. Translate Text

**Endpoint:** `POST /translate`

**Description:** Translate non-English text to English for analysis.

**Request:**
```json
{
  "text": "Eres un idiota"
}
```

**Response:**
```json
{
  "original": "Eres un idiota",
  "translated": "You are an idiot",
  "detected_language": "es"
}
```

**Note:** Requires `googletrans==4.0.0-rc1` to be installed.

---

### 3. Highlight Toxic Words

**Endpoint:** `POST /highlight_words`

**Description:** Identify potentially toxic words in the text.

**Request:**
```json
{
  "text": "You stupid idiot, shut up loser"
}
```

**Response:**
```json
{
  "toxic_words": ["stupid", "idiot", "shutup", "loser"],
  "count": 4
}
```

---

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Missing or invalid parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 500 | Internal Server Error |

---

## Rate Limiting

Currently no rate limiting is implemented. In production, consider implementing rate limiting to prevent abuse.

---

## Error Response Format

```json
{
  "error": "Error message description"
}
```

---

## Configuration

### Setting Custom API Key

1. Add to `.env` file:
```
API_KEY=your_secure_api_key_here
```

2. Or set environment variable:
```bash
export API_KEY=your_secure_api_key_here  # Linux/Mac
set API_KEY=your_secure_api_key_here     # Windows CMD
```

---

## Field Descriptions

### Toxicity Categories

- **toxic**: General toxic/offensive language
- **severe_toxic**: Very toxic/hateful language
- **obscene**: Obscene or profane content
- **threat**: Threatening language
- **insult**: Insulting language
- **identity_hate**: Attacks based on identity (race, religion, etc.)

### Other Fields

- **is_bullying**: Boolean indicating if any toxicity score >= 50%
- **severity_score**: Overall score from 0-100 (weighted average of all categories)
  - 0-30: Low severity (Green)
  - 31-70: Medium severity (Yellow)
  - 71-100: High severity (Red)

---

## Best Practices

1. **Always use HTTPS in production** to protect API keys
2. **Change default API key** before deployment
3. **Implement rate limiting** to prevent abuse
4. **Log API usage** for monitoring and debugging
5. **Handle errors gracefully** on the client side
6. **Cache results** for identical inputs to reduce load

---

## Support

For issues or questions, please refer to the main project repository.
