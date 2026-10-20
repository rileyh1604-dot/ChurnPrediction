import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn_pipeline import MODEL_PATH, NON_FEATURE_COLUMNS, SCALE_COLUMNS, TARGET, clean_data, load_data

MIN_RECALL = 0.70

df = clean_data(load_data())
x = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
y = df[TARGET].astype("int32")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

X_train_ready = pd.get_dummies(X_train, dtype="float32")
X_test_ready = pd.get_dummies(X_test, dtype="float32").reindex(columns=X_train_ready.columns, fill_value=0)

scaler = StandardScaler()
X_train_ready[SCALE_COLUMNS] = scaler.fit_transform(X_train_ready[SCALE_COLUMNS])
X_test_ready[SCALE_COLUMNS] = scaler.transform(X_test_ready[SCALE_COLUMNS])

# Hold out a validation set from the training data for choosing the threshold
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_ready, y_train, test_size=0.25, random_state=42, stratify=y_train
)

X_fit_smote, y_fit_smote = SMOTE(random_state=42).fit_resample(X_fit, y_fit)

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=30,
    random_state=42,
)
model.fit(X_fit_smote, y_fit_smote)

# Highest-precision threshold that still keeps recall >= MIN_RECALL on validation
val_probabilities = model.predict_proba(X_val)[:, 1]
candidates = []
for threshold in np.arange(0.05, 0.96, 0.01):
    predictions = (val_probabilities >= threshold).astype(int)
    if recall_score(y_val, predictions) >= MIN_RECALL:
        candidates.append((threshold, precision_score(y_val, predictions, zero_division=0)))
best_threshold = float(max(candidates, key=lambda result: result[1])[0]) if candidates else 0.5

test_probabilities = model.predict_proba(X_test_ready)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)
metrics = {
    "precision": precision_score(y_test, test_predictions),
    "recall": recall_score(y_test, test_predictions),
    "roc_auc": roc_auc_score(y_test, test_probabilities),
}

joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "feature_columns": X_train_ready.columns.tolist(),
        "threshold": best_threshold,
        "metrics": metrics,
        "test_probabilities": test_probabilities,
        "y_test": y_test.to_numpy(),
    },
    MODEL_PATH,
)

print(f"threshold: {best_threshold:.2f}")
print({name: round(value, 3) for name, value in metrics.items()})
print("Saved churn_model.joblib")
