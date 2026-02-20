import pandas as pd
import numpy as np
import joblib


def load_preprocessing_pipeline(scaler_path, feature_selector_path, label_encoder_path):
    scaler = joblib.load(scaler_path)
    feature_selector = joblib.load(feature_selector_path)
    label_encoder = joblib.load(label_encoder_path)
    
    return {
        'scaler': scaler,
        'feature_selector': feature_selector,
        'label_encoder': label_encoder
    }


def preprocess_new_data(data, scaler, feature_selector):
    data_scaled = scaler.transform(data)
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    data_selected = data_scaled[feature_selector.get_feature_names_out()]
    
    return data_selected


def predict(model, data_processed):
    predictions = model.predict(data_processed)
    probabilities = model.predict_proba(data_processed)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities
    }


def predict_with_labels(model, data_processed, label_encoder):
    pred_codes = model.predict(data_processed)
    pred_labels = label_encoder.inverse_transform(pred_codes)
    probabilities = model.predict_proba(data_processed)
    
    return {
        'predictions': pred_labels,
        'prediction_codes': pred_codes,
        'probabilities': probabilities,
        'classes': label_encoder.classes_
    }


def save_preprocessing_pipeline(scaler, feature_selector, label_encoder, 
                               scaler_path='scaler.pkl', 
                               feature_selector_path='feature_selector.pkl',
                               label_encoder_path='label_encoder.pkl'):
    joblib.dump(scaler, scaler_path, compress=9)
    joblib.dump(feature_selector, feature_selector_path, compress=9)
    joblib.dump(label_encoder, label_encoder_path, compress=9)
    print(f"Preprocessing pipeline saved")


def predict_batch(model, data, scaler, feature_selector, label_encoder=None):
    data_processed = preprocess_new_data(data, scaler, feature_selector)
    data_processed = preprocess_new_data(data, scaler, feature_selector)
    predictions = model.predict(data_processed)
    probabilities = model.predict_proba(data_processed)
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities
    }


def predict_single(model, data_row, scaler, feature_selector, label_encoder=None):
    if isinstance(data_row, dict):
        data_row = pd.DataFrame([data_row])
    elif isinstance(data_row, pd.Series):
        data_row = data_row.to_frame().T
    result = predict_batch(model, data_row, scaler, feature_selector, label_encoder)
    
    return {
        'prediction': result['predictions'][0],
        'probabilities': result['probabilities'][0]
    }
