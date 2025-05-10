from flask import Flask, render_template, request
import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to classify message
def classify_message(message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        result = classify_message(message)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

