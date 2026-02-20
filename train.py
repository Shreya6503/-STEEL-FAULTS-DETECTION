import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


def train_model(X, y, n_estimators=100, random_state=68, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=(1 - test_size), random_state=random_state
    )
    
    rfc = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=15,  # Limit tree depth to prevent overfitting
        min_samples_split=5,  # Require at least 5 samples to split
        min_samples_leaf=2,  # Require at least 2 samples in leaf nodes
        max_features='sqrt'  # Use sqrt of features for better generalization
    )
    rfc.fit(X_train, y_train)
    
    y_pred_train = rfc.predict(X_train)
    y_pred_test = rfc.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    return {
        'model': rfc,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred_test)
    }


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def save_model(model, filepath):
    joblib.dump(model, filepath, compress=9)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
