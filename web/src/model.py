import numpy as np
import pandas as pd
import pickle
import os

# Sklearn
from sklearn.metrics import classification_report
from sklearn.base import clone
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def impute(df):
    # Imputing categorical columns
    CATEGORICAL_COLS.remove('Personality')

    mode_imputer = SimpleImputer(strategy='most_frequent')
    df[CATEGORICAL_COLS] = mode_imputer.fit_transform(df[CATEGORICAL_COLS])

    # Imputing numeric colummns
    mean_imputer = SimpleImputer(strategy='mean')
    df[NUMERIC_COLS] = mean_imputer.fit_transform(df[NUMERIC_COLS])
    print("Successfully imputed data\n")

def process_data(df):
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(int)

    # Process data
    for col in CATEGORICAL_COLS:
        df[col] = df[col].map({"No": 0, "Yes": 1})

    df["Personality"] = df["Personality"].map({"Introvert": 1, "Extrovert": 0})
    print("Successfully processed data\n")

def train_model(df):
    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    best_params = {'class_weight': 'balanced',
    'objective': 'binary',
    'n_estimators': 181,
    'learning_rate': 0.038099687186893334,
    'num_leaves': 80,
    'max_depth': 9,
    'min_child_samples': 28,
    'subsample': 0.8578050266925555,
    'colsample_bytree': 0.5366587253794737,
    'reg_alpha': 3.269238332213025,
    'reg_lambda': 2.341665794508301,
    'class_weight': 'balanced',
    'objective': 'binary'}

    lgbm = LGBMClassifier(
        **best_params, verbosity=-1
    )

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )

    base_models = [
        ("lgbm", lgbm),
        ("xgb", xgb),
        ("rf", rf)
    ]

    meta_model = LogisticRegression()

    scaler = StandardScaler()

    stacked_model = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", scaler),
        ("classifier", StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            passthrough=True,
            n_jobs=-1
        ))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(stacked_model, X, y, cv=skf, scoring="accuracy").mean()
    f1 = cross_val_score(stacked_model, X, y, cv=skf, scoring="f1").mean()
    auc = cross_val_score(stacked_model, X, y, cv=skf, scoring="roc_auc").mean()

    print("Stacked Model Scores:\n")
    print(f"Accuracy: {acc}")
    print(f"F1-score: {f1}")
    print(f"ROC AUC: {auc}")

    stacked_model.fit(X, y)
    print("Successfully trained data\n")
    return stacked_model, scaler

if __name__ == "__main__":
    DATA_PATH = os.path.join("web", "src", "data.csv")
    df = pd.read_csv(DATA_PATH, index_col='id')
    CATEGORICAL_COLS = df.select_dtypes(include=['object']).columns.tolist()
    NUMERIC_COLS = df.select_dtypes(include=["float64"]).columns.tolist()
    impute(df)
    process_data(df)
    final_model, scaler = train_model(df)
    # === Save model and scaler ===
    with open("web/src/model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    with open("web/src/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
