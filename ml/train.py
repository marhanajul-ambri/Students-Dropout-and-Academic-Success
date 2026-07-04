import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("STUDENT DROPOUT MODEL TRAINING")
print("=" * 60)

# ==========================================================
# STEP 1 - Load Dataset
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_version1.csv")

print("\nLoading dataset...")
print(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names:")
for col in df.columns:
    print(f"- {col}")

# ==========================================================
# STEP 2 - Detect Target Column
# ==========================================================

possible_targets = [
    "Target",
    "target",
    "TARGET",
    "Class",
    "Status",
    "Label"
]

target_column = None

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    print("\nERROR: Target column was not found.")
    print("\nAvailable columns are:")
    print(df.columns.tolist())
    raise Exception("Please check the target column name.")

print(f"\nTarget Column Detected: {target_column}")

# ==========================================================
# STEP 3 - Prepare Features
# ==========================================================

X = df.drop(columns=[target_column])
y = df[target_column]

feature_names = X.columns.tolist()

print(f"\nNumber of Features: {len(feature_names)}")

# ==========================================================
# STEP 4 - Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ==========================================================
# STEP 5 - Train Model
# ==========================================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed successfully!")

# ==========================================================
# STEP 6 - Prediction
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# STEP 7 - Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted")
recall = recall_score(y_test, predictions, average="weighted")
f1 = f1_score(y_test, predictions, average="weighted")

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ==========================================================
# STEP 8 - Save Model
# ==========================================================

model_path = os.path.join(BASE_DIR, "student_dropout_model.pkl")
feature_path = os.path.join(BASE_DIR, "feature_names.pkl")

joblib.dump(model, model_path)
joblib.dump(feature_names, feature_path)

print("\nModel saved to:")
print(model_path)

print("\nFeature names saved to:")
print(feature_path)

# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")
print("✓ student_dropout_model.pkl")
print("✓ feature_names.pkl")