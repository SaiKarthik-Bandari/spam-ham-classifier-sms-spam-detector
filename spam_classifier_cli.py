import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Download required NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load and prepare data
def load_data():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Ham = 0, Spam = 1
    return df

# Preprocess and split data
def preprocess_data(df):
    X = df['message']
    y = df['label']
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    X_vectorized = vectorizer.fit_transform(X)
    return X_vectorized, y, vectorizer

# Train the model
def train_model(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    return model

# Save the model and vectorizer
def save_model(model, vectorizer):
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the saved model and vectorizer
def load_saved_model():
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Classify message
def classify_message(message, model, vectorizer):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

def main():
    df = load_data()
    X, y, vectorizer = preprocess_data(df)
    model = train_model(X, y)
    save_model(model, vectorizer)
    
    print("Model trained successfully!")
    print("Accuracy:", accuracy_score(y, model.predict(X)) * 100, "%")

    # Interactive loop
    print("\nInteractive Spam-Ham Classifier")
    while True:
        message = input("Enter a message to classify (type 'exit' to quit): ")
        if message.lower() == 'exit':
            break
        model, vectorizer = load_saved_model()
        result = classify_message(message, model, vectorizer)
        print(f"The message is: {result}")

if __name__ == "__main__":
    main()
