import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

print("Loading dataset...")
df = pd.read_csv("data/students_dataset.csv")

feature_cols = [c for c in df.columns if c not in ("student_id", "risk_label")]
print(f"Features ({len(feature_cols)} total):", feature_cols)

X = df[feature_cols]
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Scaled Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42)
}

results = []
best_model = None
best_acc = -1
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\nBest Model selected: {best_name} with Accuracy = {best_acc:.4f}")

# Detailed report for best model
preds_best = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, preds_best))

labels_order = ["High", "Medium", "Low"]
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds_best, labels=labels_order))

# Save the best model
joblib.dump(best_model, "app/risk_model.pkl")
print("\nSuccessfully saved best model to app/risk_model.pkl")
