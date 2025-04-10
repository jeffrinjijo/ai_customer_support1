# agents/time_estimation_agent.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Global encoders
priority_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()
category_encoder = LabelEncoder()

def train_time_estimator(csv_path, model_path="time_estimator_model.pkl"):
    # Load CSV
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df.columns = df.columns.str.strip()

    # Check required columns
    required_cols = ['Priority', 'Sentiment', 'Issue Category', 'Date of Resolution']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    # Normalize & clean inputs
    df['Priority'] = df['Priority'].astype(str).str.strip().str.upper()
    df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.upper()
    df['Issue Category'] = df['Issue Category'].astype(str).str.strip().str.upper()

    # Drop rows with missing values
    df = df.dropna(subset=required_cols)

    # Simulate resolution time if no numeric column exists
    df['Resolution Hours'] = df['Date of Resolution'].apply(lambda x: 48 if "day" in str(x).lower() else 2)

    # Encode columns
    df['Priority'] = priority_encoder.fit_transform(df['Priority'])
    df['Sentiment'] = sentiment_encoder.fit_transform(df['Sentiment'])
    df['Issue Category'] = category_encoder.fit_transform(df['Issue Category'])

    # Debug print for what was seen during training
    print("✅ Seen PRIORITIES:", priority_encoder.classes_)
    print("✅ Seen SENTIMENTS:", sentiment_encoder.classes_)
    print("✅ Seen CATEGORIES:", category_encoder.classes_)

    # Prepare X and y
    X = df[['Priority', 'Sentiment', 'Issue Category']]
    y = df['Resolution Hours']

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model + encoders
    joblib.dump((model, priority_encoder, sentiment_encoder, category_encoder), model_path)

    return model


def estimate_resolution_time(priority, sentiment, category, model_path="time_estimator_model.pkl"):
    # Normalize inputs
    priority = priority.strip().upper()
    sentiment = sentiment.strip().upper()
    category = category.strip().upper()

    # Load model and encoders
    model, pe, se, ce = joblib.load(model_path)

    # Print valid options (for debugging)
    print("\n🧠 Valid values for prediction:")
    print("  ➤ Priorities:", list(pe.classes_))
    print("  ➤ Sentiments:", list(se.classes_))
    print("  ➤ Categories:", list(ce.classes_))

    # Encode values
    p = pe.transform([priority])[0]
    s = se.transform([sentiment])[0]
    c = ce.transform([category])[0]

    # Predict
    features = [[p, s, c]]
    predicted_hours = model.predict(features)[0]

    return round(predicted_hours, 1)
