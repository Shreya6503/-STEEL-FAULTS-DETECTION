import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import DBSCAN
from collections import Counter


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def separate_features_target(df):
    X = df.iloc[:, 0:27]
    y = df.iloc[:, 27:34]
    return X, y


def scale_features(X, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return X_scaled, scaler


def select_features(X, y, k=15):
    skf = SelectKBest(score_func=chi2, k=k)
    skf.fit(X, y)
    X_selected = X[skf.get_feature_names_out()]
    return X_selected, skf


def encode_target(y_df):
    target_encoded = y_df.idxmax(axis=1)
    le = LabelEncoder()
    target_encoded = le.fit_transform(target_encoded)
    return target_encoded, le


def remove_outliers(X, y, eps=None, min_samples=7):
    if eps is None:
        # Find optimal eps value
        eps_r = 0.1
        while eps_r < 10:
            dbscan_model = DBSCAN(eps=eps_r, min_samples=min_samples).fit(X)
            if Counter(dbscan_model.labels_)[-1] < 0.1 * len(X):
                eps = eps_r
                break
            eps_r = eps_r + 0.1
    
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    outliers = X[dbscan_model.labels_ == -1]
    
    X_clean = X.drop(outliers.index, axis=0)
    y_clean = y.drop(outliers.index, axis=0)
    
    return X_clean, y_clean, dbscan_model


def preprocess_pipeline(filepath):
    df = load_data(filepath)
    df = load_data(filepath)
    X, y_raw = separate_features_target(df)
    X_scaled, scaler = scale_features(X)
    y, label_encoder = encode_target(y_raw)
    X_selected, feature_selector = select_features(X_scaled, y)
    X_clean, y_clean, dbscan_model = remove_outliers(X_selected, pd.Series(y, index=X_selected.index))
    
    return {
        'X': X_clean,
        'y': y_clean.values,
        'scaler': scaler,
        'feature_selector': feature_selector,
        'label_encoder': label_encoder,
        'dbscan_model': dbscan_model
    }
