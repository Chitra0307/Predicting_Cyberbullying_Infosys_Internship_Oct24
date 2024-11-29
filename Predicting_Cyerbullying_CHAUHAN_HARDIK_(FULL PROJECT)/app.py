from flask import Flask, request, jsonify, render_template
import joblib

import re
import contractions
from symspellpy import SymSpell, Verbosity

# Load the saved model and vectorizer
model = joblib.load('cyberbullying_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# -------------------------- Data Cleaning ------------------------------------------ #
offensive_patterns = {
    "fuck": r"\b[f|_|*][\W_]*[u|*|@|_][\W_]*[c|*|_][\W_]*[k|*|_](?:[e|*|r|_|*|s|ing]*)?\b",
    "shit": r"\b[s|$|5][\W_]*[h|*|#][\W_]*[i|!|*|1][\W_]*[t|*|+]\b",
    "bitch": r"\b[b|*|8][\W_]*[i|!|1|*][\W_]*[t|*][\W_]*[c|*][\W_]*[h|*](?:[s|*]*)?\b",
    "asshole": r"\b[a|@][\W_]*[s|$|*][\W_]*[s|$|*|5][\W_]*[h|*|#][\W_]*[o|0|*][\W_]*[l|*|1][\W_]*[e|*|3]\b",
    "cocksucker": r"\b[c|k][\W_]*[o|0][\W_]*[c|*][\W_]*[k|*][\W_]*[s|$][\W_]*[u|*][\W_]*[c|*][\W_]*[k|*][\W_]*[e|*][\W_]*[r|*]\b",
    "motherfucker": r"\b[m|*][\W_]*[o|0][\W_]*[t|*|+][\W_]*[h|*|#][\W_]*[e|*][\W_]*[r|*][\W_]*[f|ph][\W_]*[u|*|@][\W_]*[c|*|_][\W_]*[k|*][\W_]*[e|*][\W_]*[r|*]\b",
    "slut": r"\b[s|$][\W_]*[l|*][\W_]*[u|*][\W_]*[t|*]\b",
    "whore": r"\b[w|h|#][\W_]*[o|0|*][\W_]*[r|*][\W_]*[e|*]\b",
    "punk": r"\b[p|*][\W_]*[u|*][\W_]*[n|*][\W_]*[k|*]\b",
    "crap": r"\b[c|*][\W_]*[r|*][\W_]*[a|@][\W_]*[p|*]\b",
    "loser": r"\b[l|*][\W_]*[o|0|*][\W_]*[s|$][\W_]*[e|*][\W_]*[r|*]\b",
    "gay": r"\b[g|*][\W_]*[a|@][\W_]*[y|*]\b",
    "retarded": r"\b[r|*][\W_]*[e|*][\W_]*[t|*][\W_]*[a|@][\W_]*[r|*][\W_]*[d|*][\W_]*[e|*][\W_]*[d|*]\b",
    "dumb": r"\b[d|*][\W_]*[u|*][\W_]*[m|*][\W_]*[b|*|8]\b",
    "freak": r"\b[f|*][\W_]*[r|*][\W_]*[e|*][\W_]*[a|@][\W_]*[k|*]\b",
    "kill": r"\b[k|*][\W_]*[i|!][\W_]*[l|*|!][\W_]*[l|*|!]\b",
    "die": r"\b[d|*][\W_]*[i|!|1][\W_]*[e|*|3]\b",
    "hate": r"\b[h|*|#][\W_]*[a|@][\W_]*[t|*|+][\W_]*[e|*]\b",
    "ugly": r"\b[u|*][\W_]*[g|*][\W_]*[l|*][\W_]*[y|*]\b",
    "nerd": r"\b[n|*][\W_]*[e|*|3][\W_]*[r|*][\W_]*[d|*]\b",
    "sex": r"\b[s|5|$][\W_]*[e|3][\W_]*[x|*]\b",
    "dick": r"\b[d|*][\W_]*[i|!][\W_]*[c|*][\W_]*[k|*]\b",
    "fat": r"\b[f|*][\W_]*[a|@][\W_]*[t|*]\b",
    "stupid": r"\b[s|$|*][\W_]*[t|*|+][\W_]*[u|*][\W_]*[p|*][\W_]*[i|!][\W_]*[d|*]\b",
    "boobs": r"\b[b|8][\W_]*[o|0|*][\W_]*[o|0|*][\W_]*[b|8][\W_]*[s|$]\b",
    "nipple": r"\b[n|*][\W_]*[i|!|1][\W_]*[p|*][\W_]*[p|*][\W_]*[l|*|1][\W_]*[e|*]\b",
    "moron": r"\b[m|*][\W_]*[o|0][\W_]*[r|*][\W_]*[o|0|*][\W_]*[n|*]\b",
    "nigga": r"\b[n|*][\W_]*[i|1|!][\W_]*[g|*][\W_]*[g|*][\W_]*[a|@]\b",
    "cock": r"\b[c|k][\W_]*[o|0][\W_]*[c|k]\b",
    "ass": r"\b[a|@][\W_]*[s|$|*]{2}\b",
    "suck": r"\b[s|$|5][\W_]*[u|*][\W_]*[c|*][\W_]*[k|*]\b",
    "dumbass": r"\b[d|*][\W_]*[u|*][\W_]*[m|*][\W_]*[b|*][\W_]*[a|@][\W_]*[s|$|*]{2}\b",
    "dumass": r"\b[d|*][\W_]*[u|*][\W_]*[b|*][\W_]*[a|@][\W_]*[s|$|*]{2}\b",
    "bullshit": r"\b[b|8][\W_]*[u|*][\W_]*[l|*][\W_]*[l|*][\W_]*[s|$|5][\W_]*[h|*|#][\W_]*[i|!][\W_]*[t|*|+]\b",
    "cunt": r"\b[c|*][\W_]*[u|*][\W_]*[n|*][\W_]*[t|*]\b",
}

abbreviations = {
    "4ao" : "for adults only",
    "every1": "everyone",
    "fu" : "fuck you",
    "wldn’t": "would not",
    "urself": "yourself",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "insta": "instagram",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "b4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "perv" : "pervert",
    "qpsa" : "what happens",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired",
    "lov" : "love",
    "2mw" : "tomorrow",
    "r" : "are",
    "helo": "hello",
    "y" : "why",
    "ur": "your",
    "u" : "you",
    "xoxo": "hugs and kisses"
}

cyberbullying_words = {
    "noob", "dumbass", "asshole", "bitch", "slut", "whore", "idiot", "loser", "retard", "stupid",
    "fuck", "hate", "fag", "bastard", "cunt", "douchebag", "pussy", "gay", "fatass", "ugly",
    "moron", "cock", "dickhead", "scumbag", "shithead", "dick", "prick", "bitchass", "nerd", "geek",
    "jackass", "twat", "skank", "slob", "worthless", "wimp", "freak", "asswipe", "tool", "creep",
    "assclown", "dumbfuck", "douche", "shitbag", "stinker", "pansy", "weakling", "chump", "dipshit",
    "scum", "pig", "stalker", "cocksucker", "bimbo", "slutty", "hoe", "total", "asshat", "buttface", "cockroach",
    "numbnuts", "shitforbrains", "fool", "dipstick", "turd", "garbage", "manwhore", "cumslut", "suck",
    "nigger", "whimp", "blowjob", "homo", "faggot", "wuss", "coward", "vulgar", "jerk", "harlot", "dumb",
    "looser", "fucktard", "lame", "bitchy", "scummy", "bleep", "whoreish", "murderer", "gimp", "stooge",
    "simpleton", "gullible", "bastardize", "jerkoff", "wanker", "noobish", "retarded", "punk", "crap",
    "sex", "fat", "boobs", "nipple", "nigga", "ass", "dumass", "bullshit", "likes", "attack", "abuse",
}

# Convert text to lowercase
def convert_to_lowercase(text):
    """
    Convert the given text to lowercase.
    """
    return text.lower()

# Handle @, # and $ 
def process_text(text):
    """
    Replace special characters based on specific rules:
    - `$` to `s` only if followed by `l`, `1`, `!`, 'e', '*'
    - Replace any word with `<USER>` if it contains `@` followed by `s` or `$`.
    """
    # Replace `$` with `s` only when followed by `l`, `1`, `!` , 'e', '*'
    text = re.sub(r'\$(?=[l1!e*])', 's', text)

    # Replace words containing `@` followed by `s` or `$` with <USER>
    text = re.sub(r'@(?=ss|\$\$|\$s|s\$)', 'a', text)

    # Replace `#` with `h` only when followed by `a` and 't'
    text = re.sub(r'#(?=\w*(at|@+|@t|a+))', 'h', text)

    return text

# Define the function to replace masked words
def replace_masked_words(text):
    """
    Replace makked words with their actual form in the given text.
    """
    for word, pattern in offensive_patterns.items():
        text = re.sub(pattern, word, text)
    return text

# Replace URLs with a placeholder
def replace_urls(text):
    """
    Replace URL's in the given text with a placeholder <URL>.
    """
    return re.sub(r'http\S+|www\S+|https\S+', '<URL>', text)

# Define the function for abbrevations
def replace_abbreviations(text):
    """
    Replace abbreviations with their full forms in the given text.
    """
    for abbr, full_form in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)  # Replace abbreviation with full form
    return text

# Contractions
def expand_contractions(text):
    """
    Expands common contractions in the text to their full form.
    """
    return contractions.fix(text)

# Remove \n characters
def remove_newlines(text):
    """
    Remove all newline characters from the text.
    """
    return text.replace('\n', ' ')  # Replace newlines with a space (or '' to remove entirely)

# Remove punctuations
def remove_punctuation(text):
    """
    Remove all punctuation from the text except for <USER> and <URL>.
    """
    # # Regular expression to remove all punctuation except <USER> and <URL>
    text = re.sub(r'[^\w\s<URL>]', '', text)
    return text

# Remove repeated letters  (yessss -> yess)
def remove_repeated_letters(text):
    # Split text into words
    words = text.split()
    corrected_words = []

    for word in words:
        # Reduce repeated characters to 2 if there are more than 2 occurrences
        corrected_word = re.sub(r'(.)\1{2,}', r'\1\1', word)
        corrected_words.append(corrected_word)

    return ' '.join(corrected_words)

# Spell Correction with Cyberbullying Word Priority
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
sym_spell.create_dictionary_entry("noob", 11408100)      # Frequency for 'noob'
sym_spell.create_dictionary_entry("dumbass", 19501851)   # Frequency for 'dumbass'
sym_spell.create_dictionary_entry("asshole", 126713392)  # Frequency for 'asshole'

def correct_spelling_with_priority(text):
    """
    Correct spelling using SymSpell, prioritizing predefined cyberbullying words.
    """
    corrected_words = []
    
    for word in text.split():
        # Get all suggestions from SymSpell
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        
        if suggestions:
            # Get all possible corrections
            possible_words = {s.term for s in suggestions}
            
            # Check if any match a cyberbullying word
            toxic_match = next((w for w in possible_words if w in cyberbullying_words), None)
            
            if toxic_match:
                # Use the matched cyberbullying word
                corrected_words.append(toxic_match)
            else:
                # Otherwise, use the highest frequency word
                corrected_words.append(suggestions[0].term)
        else:
            # If no suggestions, keep the original word
            corrected_words.append(word)
    
    # Return the corrected sentence
    return " ".join(corrected_words)

# Remove numbers
def remove_numbers(text):
    """
    Remove numerical values from the text using regex.
    """
    return re.sub(r'\d+', '', text)

# Remove leading and trailing whitespaces and multiple spaces in the sentence
def strip_text(text):
    """
    # Replace multiple spaces with a single space and remove leading/trailing spaces
    """
    return re.sub(r'\s+', ' ', text).strip()

# Function to clean text
def clean_text(text):
    text = convert_to_lowercase(text) 
    text = process_text(text) 
    text = replace_masked_words(text) 
    text = replace_urls(text)
    text = replace_abbreviations(text) 
    text = expand_contractions(text) 
    text = remove_newlines(text)
    text = remove_repeated_letters(text)
    text = remove_punctuation(text)
    text = correct_spelling_with_priority(text)
    text = remove_numbers(text)
    text = strip_text(text) 

    return text
# ----------------------------------------------------------------------------------------

# Define your prediction function
def predict_text(text, model, vectorizer):
    # Clean the input text using your `clean_text` function
    cleaned_text = clean_text(text)
    
    # Transform the cleaned text using the vectorizer
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Get the predicted probabilities for each class
    probabilities = model.predict_proba(vectorized_text)[0]
    
    # Define the labels for each class
    labels = ['toxic', 'obscene', 'insult', 'threat', 'identity_hate']
    
    # Get the classes with probability > 0.5 (you can adjust this threshold)
    predicted_labels = [labels[i] for i in range(len(probabilities)) if probabilities[i] > 0.5]
    
    # If no labels are predicted (i.e., all probabilities <= 0.5), classify as "not_cyberbullying"
    if not predicted_labels:
        return cleaned_text,'not_cyberbullying'
    
    # Return the predicted labels along with their probabilities
    result = {label: probabilities[labels.index(label)] for label in predicted_labels}
    
    return cleaned_text , result

# Flask routes
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')   # This should point to the landing page

@app.route('/model')
def model_page():
    return render_template('model.html', prediction=None)  # Route for the model prediction page

@app.route('/faq')
def faq():
    return render_template('faq.html')    # Route for the faq page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided or invalid JSON format'}), 400
    
    user_input = data['text']
    cleaned_text, predictions = predict_text(user_input, model, vectorizer)

    if predictions == 'not_cyberbullying':
        return jsonify({'cleaned_text': cleaned_text, 'predictions': {}})
    else:
        return jsonify({'cleaned_text': cleaned_text, 'predictions': predictions})

if __name__ == "__main__":
    app.run(debug=True)