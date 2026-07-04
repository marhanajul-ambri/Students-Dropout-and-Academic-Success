import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "student_dropout_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)