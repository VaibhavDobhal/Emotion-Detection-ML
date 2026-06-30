# 🧠 Emotion Detection from Text using Machine Learning

A Machine Learning-based web application that detects the emotion expressed in a piece of text using **Natural Language Processing (NLP)** and **Machine Learning**.

The application preprocesses user input, converts it into TF-IDF features, and predicts the most likely emotion along with a confidence score. The project also compares multiple machine learning algorithms and evaluates their performance using various metrics.

---

# 🌐 Live Demo

🔗 **Website:** https://emotion-detection-ml.onrender.com

---

# 📄 Project Report

A detailed report containing dataset description, preprocessing, feature engineering, model comparison, evaluation metrics, graphs, observations, and conclusions is available in:

📁 **Project_Report/Emotion_Detection_Project_Report.pdf**

---

# ✨ Features

- Detects emotion from user-entered text
- NLP-based text preprocessing
- TF-IDF Feature Extraction
- Confidence Score Prediction
- Responsive Web Interface
- Flask Backend
- Machine Learning Integration
- Live Deployment using Render
- GitHub Version Control

---

# 🎯 Emotion Classes

The model predicts the following emotions:

- 😊 Joy
- 😢 Sad
- 😠 Anger
- 😨 Fear
- 😐 Neutral

---

# 🛠 Technologies Used

## Programming Language

- Python

## Machine Learning

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- KMeans Clustering

## Natural Language Processing

- NeatText
- TF-IDF Vectorizer

## Libraries

- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib

## Web Development

- Flask
- HTML
- CSS

## Deployment

- GitHub
- Render

---

# 📂 Project Structure

```
Emotion-Detection-ML
│
├── app.py
├── train_model.py
├── emotion_model.pkl
├── emotiondataset.csv
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
│
├── templates
│   └── index.html
│
├── static
│   └── style.css
│
├── Project_Report
│   └── Emotion_Detection_Project_Report.pdf
│
└── screenshots
    ├── home.png
    └── prediction.png
```

---

# ⚙️ Machine Learning Pipeline

```
User Input
      │
      ▼
Text Cleaning
(Remove URLs, Emojis, Stopwords, Punctuation)
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
(Logistic Regression)
      │
      ▼
Emotion Prediction
      │
      ▼
Confidence Score
```

---

# 📊 Models Evaluated

The project compares multiple Machine Learning algorithms for emotion classification.

| Model | Type |
|--------|------|
| Logistic Regression | Supervised Learning |
| Support Vector Machine (SVM) | Supervised Learning |
| Random Forest | Supervised Learning |
| KMeans Clustering | Unsupervised Learning |

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve
- PCA Visualization

---

# 📈 Results

- Successfully classified emotions from text using Machine Learning.
- Compared Logistic Regression, SVM, Random Forest, and KMeans.
- Logistic Regression and SVM achieved the best overall performance during experimentation.
- Random Forest produced competitive results but with lower consistency.
- KMeans was used for unsupervised clustering and visualization.
- The deployed web application uses **Logistic Regression** because it provides reliable deployment, native probability estimation, fast inference, and stable performance.

---

# ⚠️ Limitations

- The dataset is relatively small and may not represent all writing styles.
- The model predicts only five emotion classes (Joy, Sad, Anger, Fear, and Neutral).
- The application supports only English text.
- The model may struggle with sarcasm, irony, slang, and ambiguous sentences.
- Predictions are based only on the given sentence and do not consider previous conversation context.
- The model analyzes only text and cannot recognize emotions from speech, facial expressions, or images.
- Very short or incomplete sentences may lead to lower prediction accuracy.
- The application is deployed on Render's free tier, so the first request after inactivity may take a few seconds while the service wakes up.

---

# 🚀 Future Scope

Future improvements include:

- Improve accuracy using Transformer-based models such as BERT or RoBERTa.
- Add multilingual emotion detection.
- Integrate speech emotion recognition.
- Integrate facial emotion recognition using Computer Vision.
- Display probability scores for all emotions.
- Increase dataset size for better generalization.
- Store prediction history for users.
- Develop REST APIs for integration with other applications.
- Create Android and iOS mobile applications.
- Containerize the application using Docker and deploy on cloud platforms.

---

# 🧪 Sample Inputs

### Joy

```
I got selected for my dream company today!
```

### Sad

```
I feel lonely and depressed.
```

### Anger

```
I am extremely angry with everyone.
```

### Fear

```
Someone is following me at night.
```

### Neutral

```
Today was an ordinary day.
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/VaibhavDobhal/Emotion-Detection-ML.git
```

Move into the project folder

```bash
cd Emotion-Detection-ML
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# 👨‍💻 Author

**Vaibhav Dobhal**

B.Tech Computer Science Engineering

University of Petroleum and Energy Studies (UPES)

GitHub: https://github.com/VaibhavDobhal

---

# 🙏 Acknowledgements

- Scikit-Learn
- Flask
- Pandas
- NumPy
- NeatText
- Matplotlib
- Seaborn
- Render

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Thank you for visiting this repository!
