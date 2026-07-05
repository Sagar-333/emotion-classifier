import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

de_label_map = {
    "Angst": "fear", "Ekel": "disgust", "Freude": "joy",
    "Scham": "shame", "Schuld": "guilt", "Traurigkeit": "sadness", "Wut": "anger",
}

all_texts = []
all_labels = []

isear = pd.read_csv("ISEAR.csv", names=["emotion", "text", "empty"])
all_texts += list(isear["text"])
all_labels += list(isear["emotion"])

en = pd.read_csv("enISEAR.tsv", sep="\t")
all_texts += list(en["Sentence"])
all_labels += list(en["Prior_Emotion"].str.lower())

de2en = pd.read_csv("de2enISEAR.tsv", sep="\t")
all_texts += list(de2en["Sentence"])
all_labels += list(de2en["Prior_Emotion"].str.lower())

de = pd.read_csv("deISEAR.tsv", sep="\t")
de_labels_english = de["Prior_Emotion"].map(de_label_map)
all_texts += list(de["Sentence"])
all_labels += list(de_labels_english)

print(f"Total combined training sentences: {len(all_texts)}")

# Carve out a small 10% slice just as a rough sanity check
X_train, X_check, y_train, y_check = train_test_split(
    all_texts, all_labels, test_size=0.1, random_state=42
)

model = make_pipeline(CountVectorizer(binary=True), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)

print("\n=== Quick sanity check (not a real scientific test, just a smoke test) ===")
preds = model.predict(X_check)
print(classification_report(y_check, preds))

joblib.dump(model, "emotion_model.pkl")
print("\nSaved bilingual model -> emotion_model.pkl (your live app now uses this)")