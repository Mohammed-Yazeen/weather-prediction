import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🚀 Starting weather_prediction.py...")

try:
    # Ensure model folder exists
    if not os.path.exists('model'):
        os.makedirs('model')
        print("✅ Created 'model' folder")

    # Load dataset
    dataset_path = 'seattle-weather.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset '{dataset_path}' not found!")

    df = pd.read_csv(dataset_path)
    print("📊 Dataset loaded. First 5 rows:")
    print(df.head())

    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Features and target
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'month', 'day']]
    y = df['weather']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("🔤 Labels encoded:", list(le.classes_))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")

    # Save model and label encoder separately
    with open('model/weather_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("💾 Model and label encoder saved in 'model/' folder")

except Exception as e:
    print("⚠️ Error occurred:", e)

print("🏁 Script finished!")
