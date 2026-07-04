import pandas as pd
import joblib
from sklearn.metrics import classification_report

# 1. Load the brain we already trained and saved (no retraining here!)
model = joblib.load('emotion_model_paper.pkl')

# 2. Load the German quiz - sentences already translated to English by the
#    original researchers, with the feeling-word blanked out as "..."
test_data = pd.read_csv("de2enISEAR.tsv", sep="\t")
X_test = test_data['Sentence']
y_test = test_data['Prior_Emotion'].str.lower()

# 3. Quiz it and print the report card
predictions = model.predict(X_test)
print("\n=== RESEARCH EVALUATION REPORT (tested on German sentences, translated) ===")
print(classification_report(y_test, predictions))