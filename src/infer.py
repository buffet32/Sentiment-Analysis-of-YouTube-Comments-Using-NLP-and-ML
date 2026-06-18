# src/infer.py
import pandas as pd
import joblib
from collections import Counter

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load new comments (example: CSV with column 'text')
df = pd.read_csv("data_test.csv")
X_new = vectorizer.transform(df['text'])
preds = model.predict(X_new)

df['predicted_emotion'] = preds

# Aggregate counts
counts = Counter(preds)
total = sum(counts.values())
distribution = {k: f"{v} ({v/total:.2%})" for k,v in counts.items()}

print("Counts:", counts)
print("Distribution:", distribution)

# Save predictions
df.to_csv("data_test_with_predictions.csv", index=False)
print("Predictions saved to data_test_with_predictions.csv")
