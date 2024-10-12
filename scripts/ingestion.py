import pandas as pd

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df

if __name__ == "__main__":
    data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_data(data_path)
    print(df.head())  
