import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import os

print("🧠 Initializing Advanced Multi-Class ML Pipeline (SVM + GridSearch)...")

# 1. LOAD THE CUSTOM DATASET
try:
    df = pd.read_csv("crime_data.csv")
    print(f"Loaded {len(df)} rows from custom dataset.")
except FileNotFoundError:
    print("❌ Error: crime_data.csv not found. Run generate_data.py first!")
    exit()

df["description"] = df["description"].astype(str).str.lower().str.strip()

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["description"], df["category"], test_size=0.2, random_state=42
)

# 3. BUILD THE PIPELINE
# Using probability=True is required so we can get Confidence % in Streamlit
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english')),
    ("svm", SVC(probability=True, kernel='linear'))
])

# 4. HYPERPARAMETER TUNING
param_grid = {
    'tfidf__max_features': [2000, 5000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'svm__C': [1, 10]
}

print("⚙️ Running GridSearch to find the optimal model settings...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n✅ Tuning Complete! Best parameters: {grid_search.best_params_}")

# 5. EVALUATION
print("\n📊 --- Phase 4: Model Evaluation ---")
y_pred = best_model.predict(X_test)
target_names = ["Trace Evidence (0)", "Biological Evidence (1)", "Weaponry (2)", "Digital Evidence (3)"]
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. GENERATE CONFUSION MATRIX CHART
print("🎨 Generating Confusion Matrix chart...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.ylabel('Actual Label (Ground Truth)')
plt.xlabel('Predicted Label (AI Output)')
plt.title('CrimeMind Multi-Class Evidence Classifier')

plt.savefig("confusion_matrix_report.png")
print("✅ Chart saved as 'confusion_matrix_report.png'")

# 7. SAVE THE UPGRADED MODEL
os.makedirs("agent", exist_ok=True)
joblib.dump(best_model, "agent/clue_classifier.pkl")
print("✅ Upgraded multi-class model saved to agent/clue_classifier.pkl!")