# app.py
import streamlit as st
import joblib
import pandas as pd
import os
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="😊",
    layout="wide"
)

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
@st.cache_resource
def load_model():
    # Use absolute paths to the model files
    model_path = os.path.join(SCRIPT_DIR, "emotion_model.pkl")
    vectorizer_path = os.path.join(SCRIPT_DIR, "vectorizer.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file not found at: {vectorizer_path}")
        st.stop()
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def analyze_text(text, model, vectorizer):
    X = vectorizer.transform([text])
    emotion = model.predict(X)[0]
    return emotion

# Main app
def main():
    st.title("🎭 Emotion Analysis Tool")
    st.write("Analyze emotions in text using our AI model")
    
    # Load model
    with st.spinner('Loading model...'):
        try:
            model, vectorizer = load_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Input section
    st.header("🔍 Analyze Text")
    user_input = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze Emotion"):
        if user_input.strip():
            with st.spinner('Analyzing...'):
                try:
                    emotion = analyze_text(user_input, model, vectorizer)
                    st.success(f"Detected Emotion: **{emotion.upper()}**")
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")
        else:
            st.warning("Please enter some text to analyze")
    
    # File upload section
    st.header(" Or Upload a File")
    st.info("Upload a text file with one comment per line")
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        if st.button("Analyze File"):
            with st.spinner('Processing file...'):
                try:
                    # Read file
                    comments = uploaded_file.read().decode("utf-8").splitlines()
                    comments = [c.strip() for c in comments if c.strip()]
                    
                    if not comments:
                        st.warning("The file is empty or contains no valid text.")
                        return
                    
                    # Analyze each comment
                    results = []
                    for comment in comments:
                        try:
                            emotion = analyze_text(comment, model, vectorizer)
                            results.append({
                                'Comment': comment,
                                'Emotion': emotion
                            })
                        except Exception as e:
                            st.warning(f"Could not analyze: {comment[:50]}...")
                            continue
                    
                    if not results:
                        st.error("No comments could be analyzed.")
                        return
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(results)
                    
                    # Show results
                    st.subheader("📋 Analysis Results")
                    st.dataframe(df)
                    
                    # Show emotion distribution
                    st.subheader("📊 Emotion Distribution")
                    emotion_counts = df['Emotion'].value_counts()
                    st.bar_chart(emotion_counts)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="emotion_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    # Add some styling
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()