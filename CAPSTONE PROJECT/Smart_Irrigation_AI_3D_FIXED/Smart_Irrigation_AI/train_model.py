
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

np.random.seed(42)
n_samples = 1200

soil_moisture = np.random.uniform(10, 90, n_samples)
temperature = np.random.uniform(15, 45, n_samples)
humidity = np.random.uniform(20, 95, n_samples)
crop_stage = np.random.choice(
    ["Seedling", "Vegetative", "Flowering", "Maturity"],
    n_samples
)
recent_rainfall = np.random.uniform(0, 50, n_samples)

labels = []
for i in range(n_samples):
    score = 0

    if soil_moisture[i] < 35:
        score += 2
    elif soil_moisture[i] < 50:
        score += 1

    if temperature[i] > 32:
        score += 1

    if humidity[i] < 45:
        score += 1

    if recent_rainfall[i] < 5:
        score += 2
    elif recent_rainfall[i] < 15:
        score += 1

    if crop_stage[i] in ["Vegetative", "Flowering"]:
        score += 1

    labels.append("Yes" if score >= 3 else "No")

df = pd.DataFrame({
    "soil_moisture": soil_moisture,
    "temperature": temperature,
    "humidity": humidity,
    "crop_stage": crop_stage,
    "recent_rainfall": recent_rainfall,
    "irrigation_required": labels
})

df.to_csv("data/irrigation_data.csv", index=False)

X = df.drop(columns=["irrigation_required"])
y = df["irrigation_required"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_features = [
    "soil_moisture", "temperature", "humidity", "recent_rainfall"
]
categorical_features = ["crop_stage"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = DecisionTreeClassifier(max_depth=5, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, pos_label="Yes", zero_division=0)),
    "recall": float(recall_score(y_test, pred, pos_label="Yes", zero_division=0)),
    "f1": float(f1_score(y_test, pred, pos_label="Yes", zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "report": classification_report(y_test, pred)
}

joblib.dump(pipeline, "model/irrigation_model.pkl")

with open("model/metrics.json", "w") as f:
    import json
    json.dump(metrics, f, indent=4)

print("Training completed successfully.")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print("Model saved to model/irrigation_model.pkl")
