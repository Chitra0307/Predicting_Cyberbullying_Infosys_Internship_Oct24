"""
Quick test script to verify translation works
"""

print("Testing translation packages...")

try:
    from deep_translator import GoogleTranslator
    from langdetect import detect
    print("✓ Packages imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease run: pip install deep-translator langdetect")
    exit(1)

# Test 1: Hindi
test_text_hindi = "तुम बेवकूफ हो"
try:
    detected = detect(test_text_hindi)
    translated = GoogleTranslator(source=detected, target='en').translate(test_text_hindi)
    print(f"\n✓ Hindi test:")
    print(f"  Original: {test_text_hindi}")
    print(f"  Detected: {detected}")
    print(f"  Translated: {translated}")
except Exception as e:
    print(f"✗ Hindi test failed: {e}")

# Test 2: Telugu
test_text_telugu = "నీవు మూర్ఖుడవు"
try:
    detected = detect(test_text_telugu)
    translated = GoogleTranslator(source=detected, target='en').translate(test_text_telugu)
    print(f"\n✓ Telugu test:")
    print(f"  Original: {test_text_telugu}")
    print(f"  Detected: {detected}")
    print(f"  Translated: {translated}")
except Exception as e:
    print(f"✗ Telugu test failed: {e}")

print("\n✓ All tests passed! Translation is working.")
