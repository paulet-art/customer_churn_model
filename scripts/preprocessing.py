import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """Preprocess the data for modeling."""
    df = df.drop(columns=['customerID'])  
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    return df

if __name__ == "__main__":
    df = pd.read_csv("/home/paulet/Documents/customer_churn_model/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_processed = preprocess_data(df)
    df_processed.to_csv("data/processed/telco_churn_processed.csv", index=False)
    print("Data preprocessed and saved.")

