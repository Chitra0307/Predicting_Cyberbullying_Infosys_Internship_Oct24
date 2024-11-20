# API Keys Guide - How to Get and Set Up

## 🔑 Two API Keys Needed

Your project uses TWO different API keys:

1. **Google Gemini API Key** - For AI-powered conclusions
2. **Custom REST API Key** - For securing your API endpoint (you create this yourself)

---

## 1️⃣ Google Gemini API Key

### What It's For:
- Powers the `/conclude` endpoint
- Generates AI conclusions about cyberbullying analysis
- Example: "The analysis indicates the comment is likely cyberbullying based on high toxic and insult scores."

### How to Get It:

#### Step 1: Visit Google AI Studio
Go to: https://aistudio.google.com/app/apikey

#### Step 2: Sign In
- Use your Google account
- Accept terms of service if prompted

#### Step 3: Create API Key
- Click **"Get API Key"** or **"Create API Key"**
- Select **"Create API key in new project"** (or use existing project)
- Copy the key (starts with `AIza...`)

#### Step 4: Add to `.env` File
Open your `.env` file and add:
```
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Important Notes:
- ✅ **Free tier available** - Generous limits for testing
- ⚠️ **Keep it secret** - Never commit to GitHub
- 📊 **Monitor usage**: Check quotas at https://aistudio.google.com/

---

## 2️⃣ Custom REST API Key (For Your API Endpoint)

### What It's For:
- Secures `/api/v1/predict` endpoint
- Prevents unauthorized access to your API
- Currently set to default: `demo_api_key_12345`

### How to Create It:

#### Option 1: Generate Secure Random Key (Recommended)
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

This creates something like:
```
vJ8FkM2pQ7nL9xR4wS6tY3eH5bN0mK1c
```

#### Option 2: Create Your Own
Make it:
- At least 32 characters long
- Mix of letters, numbers, symbols
- Unique and unpredictable

Example: `MySecure_API_Key_2024_xYz123!@#`

#### Step 3: Add to `.env` File
Open your `.env` file and add/update:
```
API_KEY=vJ8FkM2pQ7nL9xR4wS6tY3eH5bN0mK1c
```

---

## 📝 Your Complete `.env` File Should Look Like:

```
# Google Gemini API for AI conclusions
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Custom API key for REST endpoint security
API_KEY=vJ8FkM2pQ7nL9xR4wS6tY3eH5bN0mK1c
```

---

## 🔒 Security Best Practices

### DO:
✅ Keep `.env` file in `.gitignore` (already done)
✅ Use different keys for development/production
✅ Rotate keys periodically
✅ Use secure random generation for keys

### DON'T:
❌ Commit `.env` to GitHub
❌ Share keys in screenshots/videos
❌ Use simple/predictable keys
❌ Hardcode keys in source code

---

## 🧪 Testing Your API Keys

### Test Google Gemini API:
1. Restart Flask: `python app.py`
2. Open http://localhost:5000
3. Enter text and click Predict
4. If conclusion shows, API key works! ✅

### Test Custom REST API Key:
```bash
# Should work with correct key:
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_CUSTOM_KEY_HERE" \
  -d "{\"comment\": \"test\"}"

# Should return 401 with wrong key:
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: wrong_key" \
  -d "{\"comment\": \"test\"}"
```

---

## ❓ Troubleshooting

### Issue: "API key not valid"
**Solution:** 
- Check for typos in `.env` file
- Ensure no extra spaces
- Verify key is active on Google AI Studio

### Issue: "Module not found: dotenv"
**Solution:**
```bash
pip install python-dotenv
```

### Issue: Changes not applying
**Solution:**
- Save `.env` file
- Restart Flask app (Ctrl+C, then `python app.py`)

---

## 💰 Pricing

### Google Gemini API:
- **Free tier**: Generous limits for development
- See current limits: https://ai.google.dev/pricing

### Your Custom API Key:
- **Free**: You create and manage it yourself
- No external service charges

---

## 🔄 Quick Setup Commands

```bash
# 1. Generate custom API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Edit .env file
notepad .env

# 3. Add both keys:
#    GOOGLE_API_KEY=AIza...
#    API_KEY=generated_key_here

# 4. Restart Flask
python app.py
```

---

## ✅ Ready to Go!

Once both keys are in `.env`:
- Your app generates AI conclusions ✅
- Your API endpoint is secured ✅
- You can test all features ✅
