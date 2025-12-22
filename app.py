from flask import Flask, request, jsonify, render_template_string
import joblib
import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/lead_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

label_map = {0: "Cold Lead", 1: "Warm Lead", 2: "Hot Lead"}

# Simple HTML page
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lead Classification</title>
</head>
<body>
    <h2>Lead Classification System</h2>
    <form method="post" action="/predict">
        <textarea name="conversation" rows="5" cols="50"
        placeholder="Paste customer conversation here"></textarea><br><br>
        <button type="submit">Predict Lead</button>
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["conversation"]
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    return render_template_string(
        HTML_PAGE,
        result=label_map[prediction]
    )

if __name__ == "__main__":
    app.run(debug=True)

