import pandas as pd
import re
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")
from nltk.corpus import stopwords

data = pd.read_csv("../data/leads.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

data["conversation"] = data["conversation"].apply(clean_text)

label_map = {"Cold": 0, "Warm": 1, "Hot": 2}
data["label"] = data["label"].map(label_map)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(data["conversation"])
y = data["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "lead_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training completed successfully")
