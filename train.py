import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

data = pd.read_csv("ISEAR.csv", names=["emotion", "text", "empty"])
X_train = data['text']
y_train = data['emotion']

test_data = pd.read_csv("enISEAR.tsv", sep="\t")
X_test = test_data['Sentence']
y_test = test_data['Prior_Emotion'].str.lower()

model = make_pipeline(CountVectorizer(binary=True), LogisticRegression(max_iter=2000))

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\n=== RESEARCH EVALUATION REPORT (tested on enISEAR) ===")
print(classification_report(y_test, predictions))
joblib.dump(model, 'emotion_model_paper.pkl')