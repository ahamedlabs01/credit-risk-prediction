import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "german.data"
AUGMENTED_DATA_PATH = BASE_DIR / "data" / "german_credit_cibil.csv"
MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.pkl"

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "dependents", "telephone", "foreign_worker", "target"
]

def generate_cibil_scores(df: pd.DataFrame, random_state: int = 42) -> pd.Series:
    """
    Synthesize realistic CIBIL bureau scores (300 to 900) based on historical financial
    stability factors (pre-loan indicators) with zero target leakage.
    """
    rng = np.random.RandomState(random_state)
    scores = []
    
    ch_map = {"A30": 70.0, "A31": 80.0, "A32": 25.0, "A33": -75.0, "A34": -135.0}
    chk_map = {"A11": -55.0, "A12": -10.0, "A13": 45.0, "A14": 20.0}
    sav_map = {"A61": -25.0, "A62": 0.0, "A63": 30.0, "A64": 50.0, "A65": 10.0}
    prop_map = {"A121": 30.0, "A122": 15.0, "A123": 0.0, "A124": -30.0}
    
    for _, row in df.iterrows():
        base = 680.0
        score = (
            base
            + ch_map.get(row.get("credit_history"), 0.0)
            + chk_map.get(row.get("checking_account"), 0.0)
            + sav_map.get(row.get("savings"), 0.0)
            + prop_map.get(row.get("property"), 0.0)
        )
        if int(row.get("existing_credits", 1)) >= 2:
            score += 15.0
        age = int(row.get("age", 30))
        if age >= 40:
            score += 15.0
        elif age <= 24:
            score -= 20.0
            
        noise = rng.normal(0, 30)
        cibil = int(np.clip(round((score + noise) / 5.0) * 5, 300, 900))
        scores.append(cibil)
        
    return pd.Series(scores, index=df.index, name="cibil_score")

def load_data(filepath=DATA_PATH):
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)
    if "cibil_score" not in df.columns:
        df["cibil_score"] = generate_cibil_scores(df)
        df.to_csv(AUGMENTED_DATA_PATH, index=False)
    return df

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

    print(f"Features: {len(categorical_columns)} categorical, {len(numerical_columns)} numerical (including cibil_score)")

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
    bad_prob = full_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test == 2, bad_prob)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Misclassification cost calculation at threshold 0.3
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
        },
        "has_cibil": True
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
