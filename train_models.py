# importing all the required modules
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Loading datasets...")
# loading and preprocess data
data_fake = pd.read_csv('fake-newzz/Datasets/Fake.csv')
data_true = pd.read_csv('fake-newzz/Datasets/True.csv')

print(f"Fake news count: {len(data_fake)}")
print(f"True news count: {len(data_true)}")

# drop the 'title' column from the individual DataFrames
#data_fake = data_fake.drop('title', axis=1)
#data_true = data_true.drop('title', axis=1)

# add class labels
data_fake["class"] = 0
data_true['class'] = 1

# merge the datasets
data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title','subject', 'date'], axis=1)

# shuffle data
data = data.sample(frac=1, random_state=42)
data.reset_index(inplace=True, drop=True)

print("\nDataset statistics:")
print(f"Total samples: {len(data)}")
print(f"Fake news percentage: {(data['class'] == 0).mean() * 100:.2f}%")
print(f"True news percentage: {(data['class'] == 1).mean() * 100:.2f}%")

def preprocess_text(text):
    # convert to lowercase
    text = text.lower()

    # remove any and all URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # join tokens back into text
    text = ' '.join(tokens)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("\nPreprocessing text...")
# preprocess text
data['text'] = data['text'].apply(preprocess_text)

# split data into train and test sets
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print("\nTraining set statistics:")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Training set class distribution: {y_train.value_counts().to_dict()}")

# vectorize text with TF-IDF
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=5,
    max_df=0.7
)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

print(f"Number of features: {x_train_vec.shape[1]}")

# train models with optimized parameters
models = {
    'nb_model': MultinomialNB(alpha=0.01),
    'lr_model': LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        solver='liblinear',
        class_weight='balanced'
    ),
    'rf_model': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
}

# train and evaluate models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # train the model
    model.fit(x_train_vec, y_train)

    # get predictions
    y_pred = model.predict(x_test_vec)

    # print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# save models and vectorizer
print("\nSaving models...")
for model_name, model in models.items():
    joblib.dump(model, f'models/{model_name}.joblib')

joblib.dump(vectorizer, 'models/vectorizer.joblib')

print("\nAll models trained and saved successfully!")