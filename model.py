import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def train_model():
    df = pd.read_excel("Students_Performance_data_set.xlsx")

    df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    df.drop_duplicates(inplace=True)

    # Feature Engineering
    df['study_effectiveness'] = df['attendance'] * df['hrs_study'] / 100
    df['academic_progress'] = df['current_cgpa'] - df['prev_sgpa']

    features = [
        'attendance',
        'hrs_study',
        'prev_sgpa',
        'study_effectiveness',
        'academic_progress'
    ]

    X = df[features]
    y = df['current_cgpa']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, df