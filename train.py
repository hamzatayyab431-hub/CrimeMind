import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

print("--- CrimeMind ML Pipeline Initiated ---")

# 1. DATA ACQUISITION & SETUP
# If you haven't downloaded a Kaggle dataset yet, this creates a starter file 
# so your code works immediately.
if not os.path.exists("crime_data.csv"):
    print("No Kaggle dataset found. Generating a starter dataset...")
    data = {
        "description": [
            "Found a bloody knife hidden near the server cooling rack.",
            "Smashed window with muddy footprints leading inside.",
            "A missing flash drive that contained standard lecture slides.",
            "A half-eaten turkey sandwich left on the front desk.",
            "Traces of a rare heat-activated toxic gas in the vents.",
            "A spilled cup of coffee near the victim's keyboard.",
            "Encrypted logs showing someone hacked the grading database.",
            "The victim's wallet is missing from his jacket."
        ],
        # 1 = Major Felony (High Severity), 0 = Misdemeanor/Irrelevant (Low Severity)
        "severity": [1, 1, 0, 0, 1, 0, 1, 0] 
    }
    pd.DataFrame(data).to_csv("crime_data.csv", index=False)

# Load the data
df = pd.read_csv("crime_data.csv")

# Clean & Preprocess
df["description"] = df["description"].astype(str).str.lower().str.strip()

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["description"], df["severity"], test_size=0.25, random_state=42
)

# 3. ALGORITHM IMPLEMENTATION (TF-IDF + Logistic Regression)
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training the Evidence Severity Analyzer...")
model.fit(X_train, y_train)

# 4. EVALUATION & METRICS (Save this output for your LaTeX report!)
print("\n--- Phase 4: Model Evaluation ---")
print(classification_report(y_test, model.predict(X_test), zero_division=0))

# 5. SAVE THE MODEL
os.makedirs("agent", exist_ok=True)
joblib.dump(model, "agent/clue_classifier.pkl")
print("\nSuccess! Model saved to agent/clue_classifier.pkl")