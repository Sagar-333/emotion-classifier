import sys
import joblib

# 1. Load the saved brain (this is instant!)
model = joblib.load('emotion_model.pkl')

# 2. Get the text passed over from Node.js
# sys.argv[1] is the first argument passed in the terminal command
text = sys.argv[1]

# 3. Make the prediction
prediction = model.predict([text])

# 4. Print it so Node.js can read the output
print(prediction[0])