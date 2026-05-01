import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

print("🧠 Initializing Advanced ML Pipeline (SVM + Hyperparameter Tuning)...")

# 1. LOAD THE CUSTOM DATASET
try:
    df = pd.read_csv("crime_data.csv")
    print(f"Loaded {len(df)} rows from custom dataset.")
except FileNotFoundError:
    print("❌ Error: crime_data.csv not found. Run generate_data.py first!")
    exit()

# Clean Data
df["description"] = df["description"].astype(str).str.lower().str.strip()

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["description"], df["severity"], test_size=0.2, random_state=42
)

# 3. BUILD THE PIPELINE (TF-IDF Vectorizer + Support Vector Machine)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english')),
    ("svm", SVC(probability=True)) # Probability=True allows us to see model confidence later if needed
])

# 4. HYPERPARAMETER TUNING (GridSearchCV)
# We give the computer a few options, and it tests all of them to find the absolute best combination.
param_grid = {
    'tfidf__max_features': [3000, 6000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'svm__C': [1, 10, 100],
    'svm__kernel': ['linear'],   # linear always wins on text classification
}

print("⚙️ Running GridSearch to find the optimal model settings... (This may take a few seconds)")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n✅ Tuning Complete! Best parameters found:\n{grid_search.best_params_}")

# 5. EVALUATION (Phase 4 of your Rubric)
print("\n📊 --- Phase 4: Model Evaluation ---")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. GENERATE CONFUSION MATRIX CHART
print("🎨 Generating Confusion Matrix chart for your LaTeX report...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Minor Clue (0)', 'Major Felony (1)'], 
            yticklabels=['Minor Clue (0)', 'Major Felony (1)'])
plt.ylabel('Actual Label (Ground Truth)')
plt.xlabel('Predicted Label (AI Output)')
plt.title('CrimeMind Evidence Classifier - Confusion Matrix')

# Save the chart as an image
plt.savefig("confusion_matrix_report.png")
print("✅ Chart saved as 'confusion_matrix_report.png'")

# 7. SAVE THE UPGRADED MODEL
import os
os.makedirs("agent", exist_ok=True)
joblib.dump(best_model, "agent/clue_classifier.pkl")
print("✅ Upgraded model saved to agent/clue_classifier.pkl! Streamlit is ready.")