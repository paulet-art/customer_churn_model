import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_churn_processed.csv")
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn']

    # Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    # Train 
    model = train_model(X_train, y_train)

    # Save model
    joblib.dump(model,"/home/paulet/Documents/customer_churn_model/models/model.pkl")
    print("Model trained and saved.")