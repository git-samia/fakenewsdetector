from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# download required NLTK data
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('running')
    word_tokenize("example text")
except LookupError as e:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

# error handling for model loading
try:
    print("Loading models...")
    # load the trained models and vectorizer
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.joblib'))
    print("Vectorizer loaded successfully")

    nb_model = joblib.load(os.path.join(models_dir, 'nb_model.joblib'))
    print("Naive Bayes model loaded successfully")

    lr_model = joblib.load(os.path.join(models_dir, 'lr_model.joblib'))
    print("Logistic Regression model loaded successfully")

    rf_model = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
    print("Random Forest model loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # convert to lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # join tokens back into text
    text = ' '.join(tokens)

    # remove any extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def output_label(n):
    if n == 0:
        return "Entered Text is Fake News!!"
    elif n == 1:
        return "Entered Text is NOT Fake News"

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        news_text = data['text']
        print(f"Received text: {news_text[:100]}...")

        if not news_text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        # preprocess the text
        processed_text = preprocess_text(news_text)
        print(f"Processed text: {processed_text[:100]}...")

        # vectorize the text
        vectorized_text = vectorizer.transform([processed_text])
        print("Text vectorized successfully")

        # get prediction and probability from RANDOM FOREST model
        rf_pred = rf_model.predict(vectorized_text)[0]
        rf_prob = rf_model.predict_proba(vectorized_text)[0]

        final_prediction = output_label(rf_pred)
        confidence = {
            'fake': float(rf_prob[0]),
            'real': float(rf_prob[1])
        }

        print(f"Prediction: {final_prediction}")
        print(f"Confidence: {confidence}")

        return jsonify({
            'final_prediction': final_prediction,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)