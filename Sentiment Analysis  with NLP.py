import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create a synthetic dataset
data = {
    "review": [
        "I love this product, it is amazing!",
        "Worst experience ever, I hate it.",
        "Very satisfied with the quality!",
        "Not worth the price at all.",
        "The customer service was excellent.",
        "I will never buy this again.",
        "This is the best thing I bought!",
        "Completely useless, waste of money.",
        "Decent product, works as expected.",
        "Terrible, I want a refund.",
        "Fantastic performance, I highly recommend it!",
        "Bad packaging, product arrived damaged.",
        "Superb quality, totally worth the money!",
        "The delivery was very late and frustrating.",
        "Amazing service, I am very happy."
    ],
    "sentiment": [
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "neutral", "negative",
        "positive", "negative", "positive", "negative", "positive"
    ]
}

df = pd.DataFrame(data)

# Display the dataset
print(df.head())

# Check sentiment distribution
sns.countplot(x=df["sentiment"])
plt.title("Sentiment Distribution")
plt.show()

import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df["cleaned_review"] = df["review"].apply(clean_text)
df.head()

X = df["cleaned_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

def predict_sentiment(review):
    review = clean_text(review)
    vector = vectorizer.transform([review])
    prediction = model.predict(vector)
    return prediction[0]

print(predict_sentiment("I really love this product!"))  # Expected: positive
print(predict_sentiment("This is the worst thing ever."))  # Expected: negative
