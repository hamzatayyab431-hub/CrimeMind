import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

print("🧠 Initializing Multi-Class Deception Detection Training...")

# 1. LOAD THE DATA
try:
    df = pd.read_csv("deception_data.csv")
    print(f"Loaded {len(df)} rows from deception_data.csv.")
except FileNotFoundError:
    print("❌ Error: deception_data.csv not found. Run generate_deception_data.py first!")
    exit()

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["statement"], df["label"], test_size=0.2, random_state=42
)

# 3. BUILD THE PIPELINE
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words=None)),
    ("rf", RandomForestClassifier(random_state=42))
])

# 4. HYPERPARAMETER TUNING
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [None, 10]
}

print("⚙️ Running GridSearch to find optimal parameters...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n✅ Tuning Complete! Best parameters: {grid_search.best_params_}")

# 5. EVALUATION
print("\n📊 --- Model Evaluation ---")
y_pred = best_model.predict(X_test)
target_names = ["Truthful (0)", "Evasive (1)", "Defensive (2)", "Fabricated (3)"]
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. GENERATE CONFUSION MATRIX CHART
print("🎨 Generating Confusion Matrix chart...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('CrimeMind Multi-Class Deception Classifier')
plt.savefig("deception_confusion_matrix.png")
print("✅ Chart saved as 'deception_confusion_matrix.png'")

# 7. SAVE THE MODEL
os.makedirs("agent", exist_ok=True)
joblib.dump(best_model, "agent/deception_model.pkl")
print("✅ Deception model saved to agent/deception_model.pkl!")
