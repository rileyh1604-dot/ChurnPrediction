from pathlib import Path

import pandas as pd

# Resolve files next to this script so it works from any working directory (e.g. Streamlit Cloud)
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco_customer_churn.xlsx"
MODEL_PATH = PROJECT_DIR / "churn_model.joblib"

TARGET = "Churn Value"
SCALE_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges"]

# Columns that are never model inputs: IDs, leakage (score/reason/label), constants, location
NON_FEATURE_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "Churn Reason", "CLTV", TARGET,
]


def load_data(path=DATA_PATH):
    return pd.read_excel(path)


def clean_data(df):
    # Same cleaning steps as the notebook
    df = df.copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df.loc[df["Tenure Months"] == 0, "Total Charges"] = 0
    return df


def preprocess(df, feature_columns, scaler):
    # Turn raw customer rows into the exact matrix the model was trained on
    X = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    X = pd.get_dummies(X, dtype="float32")
    X = X.reindex(columns=feature_columns, fill_value=0)
    X[SCALE_COLUMNS] = scaler.transform(X[SCALE_COLUMNS])
    return X
