import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def zeros_to_nan(X):
    X = X.copy()
    X[ZERO_COLS] = X[ZERO_COLS].replace(0, np.nan)
    return X

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip columns to [Q1 - k*IQR, Q3 + k*IQR] learned on the training data."""
    def __init__(self, cols=None, k=1.5):
        self.cols = cols; self.k = k
    def fit(self, X, y=None):
        self.bounds_ = {}
        for c in self.cols:
            q1, q3 = X[c].quantile([0.25, 0.75]); iqr = q3 - q1
            self.bounds_[c] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self
    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X

def add_features(X):
    X = X.copy()
    X["BMI_Category"] = pd.cut(X["BMI"], [0, 18.5, 25, 30, 100],
                               labels=["Underweight", "Normal", "Overweight", "Obese"]).astype(str)
    X["Age_Group"] = pd.cut(X["Age"], [0, 30, 45, 120], labels=["Young", "Middle", "Senior"]).astype(str)
    X["Glucose_BMI"] = X["Glucose"] * X["BMI"] / 100
    X["Glucose_Age"] = X["Glucose"] * X["Age"] / 100
    return X
