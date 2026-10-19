import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "german.data"
MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.pkl"

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "dependents", "telephone", "foreign_worker", "target"
]

def load_data(filepath=DATA_PATH):
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)

def train_and_save():
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    # Stratified train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_columns = X.select_dtypes(
        include=["str", "object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Preprocessor for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numerical", "passthrough", numerical_columns)
        ]
    )

    # Complete End-to-End Pipeline: Preprocessor -> Scaler -> Classifier
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    print("Fitting model...")
    full_pipeline.fit(X_train, y_train)

    y_pred = full_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Misclassification cost calculation at threshold 0.3
    bad_prob = full_pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.3
    predictions = (bad_prob >= threshold).astype(int) + 1
    cm = confusion_matrix(y_test, predictions, labels=[1, 2])
    cost = cm[1, 0] * 5 + cm[0, 1]
    print(f"Confusion Matrix (threshold={threshold}):\n{cm}")
    print(f"Total misclassification cost: {cost}")

    # Package model artifact
    final_model = {
        "model": full_pipeline,
        "threshold": threshold,
        "target_mapping": {
            1: "Good Credit",
            2: "Bad Credit"
        }
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\nTrained model successfully saved to {MODEL_PATH}!")

    # Verify prediction with a sample applicant (like app.py)
    sample_applicant = pd.DataFrame([X.iloc[0].to_dict()])
    proba = full_pipeline.predict_proba(sample_applicant)
    print(f"Verification test prediction probability: {proba}")

if __name__ == "__main__":
    train_and_save()
