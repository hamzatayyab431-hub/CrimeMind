import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob

# ─── 1. LOAD CLASSIFIER ───────────────────────────────────────────
def load_classifier():
    try:
        return joblib.load("agent/clue_classifier.pkl")
    except:
        return None

# ─── 1b. LOAD DECEPTION MODEL ─────────────────────────────────────
def load_deception_model():
    try:
        return joblib.load("agent/deception_model.pkl")
    except:
        return None

# ─── 2. CLUE SEVERITY (existing) ──────────────────────────────────
def classify_clue(model, text):
    text_lower = text.lower()
    
    # Expanded Category-Based Keyword Mapping
    crime_keywords = {
        "HOMICIDE": ["murder", "kill", "death", "dead", "body", "corpse", "slay", "strangle", "stab", "shoot", "poison", "murderer"],
        "THEFT": ["rob", "steal", "heist", "theft", "burglar", "loot", "stole", "shoplift", "snatch", "bank", "jewelry", "vault"],
        "VIOLENCE": ["assault", "hit", "punch", "fight", "attack", "wound", "blood", "weapon", "gun", "knife", "blade", "threat"],
        "SUBSTANCE": ["drug", "cocaine", "heroin", "pills", "smuggle", "illegal", "trafficking"],
        "FRAUD": ["scam", "bribe", "fraud", "forge", "falsify", "blackmail", "extort", "ransom"]
    }
    
    matches = 0
    found_categories = []
    
    for category, keywords in crime_keywords.items():
        if any(word in text_lower for word in keywords):
            matches += 1
            found_categories.append(category)
            
    # If explicit keywords found, override model
    if matches >= 2:
        return f"🚨 MAJOR FELONY ({', '.join(found_categories[:2])})", 99.9
    elif matches == 1:
        return f"🔍 CRIME RELATED: {found_categories[0]}", 95.5

    if model is not None:
        proba = model.predict_proba([text_lower])[0]
        label = int(np.argmax(proba))
        confidence = round(proba[label] * 100, 1)
        
        if label == 1:
            severity = "🚨 Biological Evidence"
        elif label == 2:
            severity = "🔫 Weaponry/Violence"
        elif label == 3:
            severity = "💻 Digital Evidence"
        else:
            severity = "📝 Trace Evidence"
            
        return severity, confidence

    # Fallback if no model and no keywords
    blob = TextBlob(text)
    polarity_intensity = abs(blob.sentiment.polarity)
    if polarity_intensity > 0.6:
        return "⚠️ HIGHLY SUSPICIOUS", 85.0
    
    return "📝 Minor Detail", 45.0

# ─── 3. SENTIMENT ANALYSIS ────────────────────────────────────────
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity      # -1 to +1

    # Refined Sentiment Thresholds
    if polarity < -0.45:
        tone = "😠 Aggressive / Hostile"
    elif polarity < -0.15:
        tone = "😟 Uneasy / Nervous"
    elif polarity > 0.45:
        tone = "😊 Calm / Cooperative"
    elif polarity > 0.15:
        tone = "🤔 Friendly / Helpful"
    else:
        # Use subjectivity for neutral statements
        if blob.sentiment.subjectivity > 0.6:
            tone = "🤨 Defensive / Evasive"
        else:
            tone = "😐 Neutral / Calculated"

    return {
        "tone": tone,
        "polarity": round(polarity, 2),
        "subjectivity": round(blob.sentiment.subjectivity, 2)
    }

# ─── 3b. DECEPTION ANALYSIS ───────────────────────────────────────
def analyze_deception(model, text):
    text_lower = text.lower()
    confession_words = ["kill", "murder", "stab", "shoot", "stole", "robbed", "did it", "guilty", "confess"]
    
    if any(word in text_lower for word in confession_words) and "i " in text_lower:
        return "🚨 Direct Confession", 99.9

    if model is None:
        return "Unknown", 0.0
    
    proba = model.predict_proba([text])[0]
    label = int(np.argmax(proba))
    confidence = round(proba[label] * 100, 1)
    
    # If confidence is extremely low, mark it as inconclusive
    if confidence < 40.0:
        return "🤷 Inconclusive / Vague", confidence
    
    if label == 1:
        return "🤔 Evasive", confidence
    elif label == 2:
        return "😠 Defensive / Hostile", confidence
    elif label == 3:
        return "🤥 Fabricated / Over-justified", confidence
    else:
        return "✅ Likely Truthful", confidence

# ─── 4. SUSPECT RANKING ───────────────────────────────────────────
def update_suspect_scores(suspects, found_clues, case_data):
    """
    Score each suspect based on how many discovered clues
    point toward them. Returns a ranked pandas DataFrame.
    """
    scores = {s["name"]: 0 for s in suspects}

    for clue in found_clues:
        clue_lower = clue.lower()
        for suspect in suspects:
            name_lower = suspect["name"].lower().split()[-1]  # last name
            motive_words = suspect["motive"].lower().split()
            # Bump score if clue mentions suspect or motive keywords
            if name_lower in clue_lower:
                scores[suspect["name"]] += 30
            for word in motive_words:
                if len(word) > 4 and word in clue_lower:
                    scores[suspect["name"]] += 10

    # Normalize to 100
    max_score = max(scores.values()) if max(scores.values()) > 0 else 1
    normalized = {k: round((v / max_score) * 100) for k, v in scores.items()}

    df = pd.DataFrame([
        {"Suspect": name, "Suspicion Score": score}
        for name, score in normalized.items()
    ]).sort_values("Suspicion Score", ascending=False).reset_index(drop=True)

    return df

# ─── 5. CLUE CLUSTERING ───────────────────────────────────────────
def cluster_clues(clues_list, n_clusters=2):
    """
    Groups discovered clues into clusters using KMeans.
    Returns clues with their cluster label.
    """
    if len(clues_list) < n_clusters:
        return None  # not enough clues yet

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(clues_list)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    cluster_names = ["🔴 Crime Scene Clues", "🔵 Suspect Behaviour Clues"]
    result = []
    for clue, label in zip(clues_list, labels):
        result.append({"clue": clue, "group": cluster_names[label % len(cluster_names)]})

    return pd.DataFrame(result)