# src/train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib  # to save your model

# 1. Load dataset
df = pd.read_csv("data_train.csv")  # make sure this file exists
print("Columns found:", df.columns)

# 2. Split into features and labels
X = df['text']
y = df['label']

# 3. Convert text to numeric features
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# 4. Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 6. Evaluate
print("Train accuracy:", model.score(X_train, y_train))
print("Validation accuracy:", model.score(X_val, y_val))

# 7. Save the model and vectorizer for later inference
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved!")
