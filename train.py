import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 1. Load the OLD textbook (original ISEAR) - study ALL of it this time
data = pd.read_csv("ISEAR.csv", names=["emotion", "text", "empty"])
X_train = data['text']
y_train = data['emotion']

# 2. Load the NEW textbook (enISEAR) - sentences the model has NEVER seen,
#    with the feeling-word blanked out as "..."
test_data = pd.read_csv("enISEAR.tsv", sep="\t")
X_test = test_data['Sentence']
y_test = test_data['Prior_Emotion'].str.lower()

# 3. Create the machine learning pipeline (this matches the paper's recipe:
#    "does this word appear, yes/no" features + a MaxEnt-style classifier)
model = make_pipeline(CountVectorizer(binary=True), LogisticRegression(max_iter=2000))

# 4. Train the model on ALL of the old textbook
model.fit(X_train, y_train)

# 5. Test it on the NEW textbook and print the research report
predictions = model.predict(X_test)
print("\n=== RESEARCH EVALUATION REPORT (tested on enISEAR) ===")
print(classification_report(y_test, predictions))
joblib.dump(model, 'emotion_model_paper.pkl')