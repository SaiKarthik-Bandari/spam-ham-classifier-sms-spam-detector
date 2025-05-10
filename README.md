# 📧 Spam-Ham Classifier (Web App using Flask)

A beginner-friendly project that teaches you how to build your own **Spam vs. Ham Message Classifier** using **Python, Machine Learning, and Flask**. This project classifies whether a given text message is spam (unwanted) or ham (legitimate). You'll also learn how to make a simple web app for it.

---

## ✨ What is This Project?

Have you ever received unwanted messages like "You've won a lottery!"? These are called **Spam** messages. Messages from your friends or work are **Ham** (not spam).

This project helps a computer understand how to **automatically detect if a message is spam or ham** using machine learning.

---

## ✅ Key Features

* 🔍 Predict whether a message is Spam or Ham
* 💡 Simple to understand and use
* 🌐 Interactive web interface using Flask
* 🧠 Trained using real-world SMS spam data
* 🔰 Made for beginners and kids who are new to ML and Python

---

## 🧠 How it Works (Simple Explanation)

1. **Collect Data** – We use a dataset of messages labeled as spam or ham.
2. **Clean the Data** – Remove unwanted words, symbols, and stopwords.
3. **Train the Model** – Use a machine learning algorithm to learn the patterns.
4. **Save the Model** – Save it to use in our app without training every time.
5. **Build a Web App** – Create a website to type a message and get results.

---

## 🧰 Tools & Technologies Used

| Technology    | Purpose                    |
| ------------- | -------------------------- |
| Python        | Programming language       |
| NLTK          | For stopword removal       |
| Sklearn       | For machine learning model |
| Flask         | For web application        |
| HTML/CSS      | For frontend design        |
| Pickle/Joblib | Save and load models       |

---

## 📦 Project Structure

```
spam_classifier_web/
│
├── app.py                  # Flask app logic
├── spam_classifier_model.pkl  # Trained ML model
├── vectorizer.pkl             # Word vectorizer
├── templates/
│   └── index.html          # HTML form
└── README.md               # This file
```

---

## 🚀 How to Run the Project (For Beginners)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/spam-ham-classifier.git
cd spam-ham-classifier
```

### Step 2: Install Requirements

Make sure Python is installed. Then run:

```bash
pip install flask nltk scikit-learn pandas
```

### Step 3: Train the Model (If Not Already Done)

You can use this code to train the model:

```python
# train_model.py (create this file)

import pandas as pd
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', sep='\t', header=None, names=['label', 'message'])
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

stop_words = set(stopwords.words('english'))

vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
```

Run:

```bash
python train_model.py
```

This creates:

* `spam_classifier_model.pkl`
* `vectorizer.pkl`

### Step 4: Create the Web App

**app.py**

```python
from flask import Flask, render_template, request
import joblib
import nltk
nltk.download('stopwords')

app = Flask(__name__)
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        prediction = model.predict(data)
        result = "Spam" if prediction[0] == 1 else "Ham"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
```

**templates/index.html**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Spam-Ham Classifier</title>
</head>
<body>
    <h2>Enter a Message:</h2>
    <form method="POST">
        <textarea name="message" rows="5" cols="40"></textarea><br><br>
        <button type="submit">Classify</button>
    </form>

    {% if result %}
        <h3>Result: {{ result }}</h3>
    {% endif %}
</body>
</html>
```

### Step 5: Run the Web App

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to use the app!

---

## 📚 Use Cases

* 📱 Detect spam messages in messaging apps
* 📩 Email spam filtering
* 🧑‍🏫 Machine learning education project
* 🎓 Great for students’ portfolios or hackathons

---

## 🌍 Want to Publish Online?

Try free platforms like:

* [Render](https://render.com)
* [Replit](https://replit.com)
* [PythonAnywhere](https://www.pythonanywhere.com/)

---

## 🙋 FAQ

**Q: Do I need machine learning knowledge?**
A: No, this project teaches you the basics as you build.

**Q: Can I modify the design?**
A: Yes! You can add CSS or use frameworks like Bootstrap.

**Q: Can I add more features?**
A: Absolutely! You can add charts, graphs, and even confidence scores.

---

## 💖 Credits

* Dataset by [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
* Inspired by Python & ML tutorials

---

## 📌 Final Notes

This is a perfect project to start learning about **machine learning, text processing, and web development** — all in one! Keep experimenting and add your own creative features. 💡


