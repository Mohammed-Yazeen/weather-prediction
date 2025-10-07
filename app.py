import pickle
import numpy as np
import pandas as pd

# Load model and label encoder
with open('model/weather_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print("🌤️ Weather Prediction App 🌤️")
print("Enter the following weather details:")

precipitation = float(input("Precipitation: "))
temp_max = float(input("Max Temperature: "))
temp_min = float(input("Min Temperature: "))
wind = float(input("Wind: "))
month = int(input("Month (1-12): "))
day = int(input("Day (1-31): "))

# Prepare input

feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind', 'month', 'day']
features = pd.DataFrame([[precipitation, temp_max, temp_min, wind, month, day]], columns=feature_names)

# Predict
prediction = model.predict(features)[0]
predicted_label = le.inverse_transform([prediction])[0]

print(f"\n🌦️ Predicted Weather Type: {predicted_label}")
