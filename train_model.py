import pandas as pd
import neattext.functions as nfx
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("emotiondataset.csv", encoding="ISO-8859-1")

# Remove missing values
df = df.dropna(subset=["Emotion "])

# Convert text to string
df["Text"] = df["Text"].astype(str)

# Text Cleaning
df["Clean_Text"] = (
    df["Text"]
    .apply(nfx.remove_userhandles)
    .apply(nfx.remove_urls)
    .apply(nfx.remove_stopwords)
    .apply(nfx.remove_punctuations)
    .apply(nfx.remove_emojis)
    .apply(nfx.remove_special_characters)
)

# Features and Labels
X = df["Clean_Text"]
y = df["Emotion "]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# SVM Model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", SVC(kernel="rbf", C=10, probability=True))
])

# Train
model.fit(X_train, y_train)

# Test Accuracy
y_pred = model.predict(X_test)

print("=" * 40)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("=" * 40)

print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "emotion_model.pkl")

print("\nModel saved successfully as emotion_model.pkl")