import os
from preprocess import preprocess_pipeline
from train import train_model, save_model
import joblib

os.makedirs('models', exist_ok=True)

print("Starting Steel Fault Detection Pipeline...")
print("\n1. Preprocessing data...")
preprocessed = preprocess_pipeline('Data/Faults.csv')
X = preprocessed['X']
y = preprocessed['y']
scaler = preprocessed['scaler']
feature_selector = preprocessed['feature_selector']
label_encoder = preprocessed['label_encoder']

print(f"   Data shape: {X.shape}")
print(f"   Target shape: {y.shape}")

print("\n2. Training model...")
results = train_model(X, y)
model = results['model']
print(f"   Training accuracy: {results['train_accuracy']:.4f}")
print(f"   Testing accuracy: {results['test_accuracy']:.4f}")

print("\n3. Saving model...")
save_model(model, 'models/steel_fault_model.pkl')
print("   Model saved successfully!")

print("\n4. Pipeline Complete! ✓")
print("   Model location: models/steel_fault_model.pkl")
