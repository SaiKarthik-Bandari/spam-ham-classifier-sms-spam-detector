import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Map label to numeric
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    words = text.split()
    filtered = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return ' '.join(filtered)

df['cleaned'] = df['message'].apply(preprocess_text)

# Visualizations
plt.figure(figsize=(12, 6))
wordcloud_spam = WordCloud(width=600, height=400).generate(' '.join(df[df['label_num'] == 1]['cleaned']))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Spam Messages")
plt.show()

plt.figure(figsize=(12, 6))
wordcloud_ham = WordCloud(width=600, height=400).generate(' '.join(df[df['label_num'] == 0]['cleaned']))
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Ham Messages")
plt.show()

# Countplot
sns.countplot(data=df, x='label')
plt.title("Distribution of Ham vs Spam")
plt.show()

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned']).toarray()
y = df['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"📈 Accuracy: {acc*100:.2f}%\n")

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred))

# Prediction
def predict_message(msg):
    msg_cleaned = preprocess_text(msg)
    msg_vector = tfidf.transform([msg_cleaned])
    prediction = model.predict(msg_vector)[0]
    return "Spam" if prediction else "Ham"

# Test predictions
print("\n🔍 Test Predictions:")
sample_msgs = [
    "Congratulations! You've won a free iPhone. Click to claim now.",
    "Hey, are we still meeting at 5?",
    "URGENT! Your account has been suspended. Verify now!",
    "Let's catch up tomorrow for lunch."
]

for msg in sample_msgs:
    print(f"> '{msg}' => {predict_message(msg)}")
