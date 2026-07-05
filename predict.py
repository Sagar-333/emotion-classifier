import sys
import joblib

model = joblib.load('emotion_model.pkl')

text = sys.argv[1]
prediction = model.predict([text])

print(prediction[0])