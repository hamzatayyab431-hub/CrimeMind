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

# ─── 2. CLUE SEVERITY (existing) ──────────────────────────────────
def classify_clue(model, text):
    if model is not None:
        proba = model.predict_proba([text.lower()])[0]
        label = int(np.argmax(proba))
        confidence = round(proba[label] * 100, 1)
        severity = "🚨 MAJOR FELONY" if label == 1 else "📝 Minor Clue"
        return severity, confidence
        
    # Expanded Category-Based Keyword Mapping
    crime_keywords = {
        "HOMICIDE": ["murder", "kill", "death", "dead", "body", "corpse", "slay", "strangle", "stab", "shoot", "poison", "murderer"],
        "THEFT": ["rob", "steal", "heist", "theft", "burglar", "loot", "stole", "shoplift", "snatch", "bank", "jewelry", "vault"],
        "VIOLENCE": ["assault", "hit", "punch", "fight", "attack", "wound", "blood", "weapon", "gun", "knife", "blade", "threat"],
        "SUBSTANCE": ["drug", "cocaine", "heroin", "pills", "smuggle", "illegal", "trafficking"],
        "FRAUD": ["scam", "bribe", "fraud", "forge", "falsify", "blackmail", "extort", "ransom"]
    }
    
    text_lower = text.lower()
    matches = 0
    found_categories = []
    
    for category, keywords in crime_keywords.items():
        if any(word in text_lower for word in keywords):
            matches += 1
            found_categories.append(category)
    
    blob = TextBlob(text)
    polarity_intensity = abs(blob.sentiment.polarity)
    
    # Dynamic confidence based on keyword density, emotional intensity, and statement length
    word_count = len(text.split())
    length_bonus = min(word_count * 0.8, 25.0)  # Up to 25% bonus for detailed statements
    random_jitter = np.random.uniform(-4.5, 8.5) # Add slight realism jitter
    
    base_confidence = 15.0 + length_bonus + (matches * 22.0) + (polarity_intensity * 18.0) + random_jitter
    confidence = min(max(round(base_confidence, 1), 12.5), 99.9)
    
    if matches >= 2:
        severity = f"🚨 MAJOR FELONY ({', '.join(found_categories[:2])})"
    elif matches == 1:
        severity = f"🔍 CRIME RELATED: {found_categories[0]}"
    elif polarity_intensity > 0.6:
        severity = "⚠️ HIGHLY SUSPICIOUS"
    else:
        severity = "📝 Minor Detail"
        
    return severity, confidence

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