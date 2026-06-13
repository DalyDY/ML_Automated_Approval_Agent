import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib


# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("final_data.csv")

print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["decision"].value_counts())


# ==========================
# Features and Target
# ==========================
X = df.drop("decision", axis=1)
y = df["decision"]

print("\nColumns Used For Training:")
print(X.columns.tolist())


# ==========================
# Encode Target
# ==========================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nClass Mapping:")
for class_name, encoded in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(f"{class_name} -> {encoded}")


# ==========================
# Identify Column Types
# ==========================
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns

print(f"\nNumeric Features: {len(numeric_features)}")
print(f"Categorical Features: {len(categorical_features)}")


# ==========================
# Preprocessing
# ==========================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================
# XGBoost Model
# ==========================
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42
            )
        )
    ]
)


# ==========================
# Train Model
# ==========================
model.fit(X_train, y_train)


# ==========================
# Predictions
# ==========================
y_pred = model.predict(X_test)


# ==========================
# Evaluation
# ==========================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ==========================
# Save Model
# ==========================
joblib.dump(
    model,
    "xgboost_model.pkl"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print("\nModel saved as xgboost_model.pkl")
print("Label encoder saved as label_encoder.pkl")