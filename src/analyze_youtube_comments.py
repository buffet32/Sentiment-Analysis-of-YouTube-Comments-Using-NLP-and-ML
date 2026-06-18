# src/analyze_youtube_comments.py
import joblib
import pandas as pd
from collections import Counter

def load_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [c.strip() for c in content.split('---') if c.strip()]

def analyze_comments():
    print("Loading model and comments...")
    model = joblib.load("emotion_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    comments = load_comments("youtube_comments.txt")
    comments = [c.split(':', 1)[1].strip() if ':' in c else c for c in comments]
    
    print("Analyzing emotions...")
    X = vectorizer.transform(comments)
    predictions = model.predict(X)
    
    results = pd.DataFrame({
        'comment': comments,
        'emotion': predictions
    })
    
    emotion_counts = Counter(predictions)
    total = len(predictions)
    
    print("\n=== Results ===")
    print(f"Total comments: {total}")
    print("\nEmotions found:")
    for emotion, count in emotion_counts.most_common():
        print(f"- {emotion}: {count} ({count/total:.1%})")
    
    results.to_csv("youtube_emotion_results.csv", index=False, encoding='utf-8')
    print("\nDetailed results saved to 'youtube_emotion_results.csv'")

if __name__ == "__main__":
    analyze_comments()