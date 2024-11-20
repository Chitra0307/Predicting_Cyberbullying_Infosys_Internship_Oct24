# Translation Feature - Important Notes

## ✅ What Works

The translation feature works for **native scripts**:

### Supported Languages (Native Scripts):
- **Telugu**: నీవు మూర్ఖుడవు → "You are a fool"
- **Hindi**: तुम बेवकूफ हो → "You are stupid"  
- **Tamil**: நீ முட்டாள் → "You are a fool"
- **Kannada**: ನೀನು ಮೂರ್ಖ → "You are a fool"
- **Bengali**: তুমি বোকা → "You are stupid"
- **Gujarati**: તમે મૂર્ખ છો → "You are foolish"

---

## ❌ What Doesn't Work

**Transliteration** (Indian language words written in English letters):

### Does NOT work:
- ❌ "neevu moorkudavu" (Telugu in English script)
- ❌ "tum bewakoof ho" (Hindi in English script)
- ❌ "nee muttaal" (Tamil in English script)

### Why?
When you write Indian language words using English letters, the language detector sees them as English text and won't translate.

---

## 💡 Best Practice

### For Translation to Work:
1. **Use native keyboard** on your device
2. **Type in Telugu/Hindi/Tamil script**
3. Click "🌐 Translate to English"

### Example Workflow:
```
Input: నీవు మూర్ఖుడవు (Telugu script)
↓
Click: Translate to English
↓
Output: You are a fool (English)
↓
Click: Predict
↓
Result: Toxicity scores
```

---

## Alternative: Use English Directly

If you can't type in native script, just type in English:
```
Instead of: "neevu moorkudavu"
Just type: "You are a fool"
```

The model analyzes English text directly, so you don't need translation.

---

## How to Type in Native Scripts

### On Android/iOS:
1. Install Gboard or SwiftKey keyboard
2. Add Telugu/Hindi/Tamil language
3. Switch keyboard when typing

### On Windows:
1. Settings → Time & Language → Language
2. Add Telugu/Hindi/Tamil  
3. Use Windows+Space to switch keyboard

### On Mac:
1. System Preferences → Keyboard → Input Sources
2. Add Telugu/Hindi/Tamil
3. Use Cmd+Space to switch

---

## Summary

✅ **Works**: Native scripts (తెలుగు, हिंदी, தமிழ்)
❌ **Doesn't Work**: Transliteration (telugu, hindi, tamil in English letters)
💡 **Solution**: Use native keyboard OR type directly in English
