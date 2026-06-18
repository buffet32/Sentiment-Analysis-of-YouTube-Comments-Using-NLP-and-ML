# Sentiment Analysis of YouTube Comments Using NLP and Machine Learning

## Overview

This project performs sentiment analysis on YouTube comments using Natural Language Processing (NLP) and Machine Learning techniques.

The system allows users to:

- Collect comments from YouTube videos using the YouTube Data API.
- Clean and preprocess textual data.
- Train a sentiment classification model.
- Predict the sentiment of new comments.
- Analyze audience opinions through an interactive interface.

## Features

### YouTube Comment Collection

- Fetch comments from YouTube videos using the YouTube Data API.
- Store collected comments for further processing.

### Text Preprocessing

- Lowercase conversion
- Removal of punctuation
- Stop-word removal
- Text cleaning and normalization

### Machine Learning Pipeline

- Feature extraction using TF-IDF Vectorization.
- Model training on labeled sentiment data.
- Sentiment prediction for unseen comments.

### Sentiment Classification

The model classifies comments into sentiment categories such as:

- Positive
- Negative
- Neutral

### Interactive Application

A simple application interface allows users to:

- Enter comments manually.
- Analyze sentiment instantly.
- View prediction results.

---

## Project Structure

```text
Sentiment-Analysis-of-YouTube-Comments-Using-NLP-and-ML/
│
├── src/
│   ├── collect_youtube.py            # Collect comments from YouTube API
│   ├── preprocess.py                 # Data preprocessing
│   ├── train.py                      # Model training
│   ├── infer.py                      # Sentiment prediction
│   ├── analyze_youtube_comments.py   # Full analysis pipeline
│   ├── app.py                        # Application entry point
│   │
│   ├── dataset.csv
│   ├── data_train.csv
│   ├── data_val.csv
│   ├── data_test.csv
│   ├── data_test_with_predictions.csv
│   │
│   ├── emotion_model.pkl
│   ├── vectorizer.pkl
│   │
│   ├── youtube_comments.txt
│   └── youtube_emotion_results.csv
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- YouTube Data API
- Pickle
- Flask / Streamlit (if applicable)

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Sentiment-Analysis-of-YouTube-Comments-Using-NLP-and-ML.git

cd Sentiment-Analysis-of-YouTube-Comments-Using-NLP-and-ML
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

```bash
python src/train.py
```

This will generate:

- emotion_model.pkl
- vectorizer.pkl

---

## Collect YouTube Comments

```bash
python src/collect_youtube.py
```

Provide:

- YouTube API Key
- Video ID

The comments will be saved locally.

---

## Run Sentiment Prediction

```bash
python src/infer.py
```

Example:

```text
Input:
I really enjoyed this video!

Output:
Positive
```

---

## Run Full Analysis

```bash
python src/analyze_youtube_comments.py
```

Output:

```text
Total Comments: 500
Positive: 65%
Negative: 20%
Neutral: 15%
```

Results are exported to:

```text
youtube_emotion_results.csv
```

---

## Example Workflow

```text
YouTube Video
      │
      ▼
Comment Collection
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
      │
      ▼
Sentiment Prediction
      │
      ▼
Results & Insights
```

---

## Future Improvements

- BERT-based sentiment analysis
- Multilingual support
- Real-time YouTube monitoring
- Sentiment dashboard
- Explainable AI using SHAP
- RAG-powered opinion analysis

---

## Author

**Yassine Nait Ider**

Artificial Intelligence & Data Science Engineering Student

Marrakech, Morocco
