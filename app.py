from flask import Flask, render_template, request
import joblib
import neattext.functions as nfx

app = Flask(__name__)

# Load the trained model
model = joblib.load("emotion_model.pkl")


def clean_text(text):
    text = nfx.remove_userhandles(text)
    text = nfx.remove_urls(text)
    text = nfx.remove_stopwords(text)
    text = nfx.remove_punctuations(text)
    text = nfx.remove_emojis(text)
    text = nfx.remove_special_characters(text)
    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    cleaned = clean_text(text)

    prediction = model.predict([cleaned])[0]

    probability = model.predict_proba([cleaned])[0]

    confidence = round(max(probability) * 100, 2)

    return render_template(
        "index.html",
        text=text,
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)