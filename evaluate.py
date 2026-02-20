import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_pipeline
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_with_cross_validation(X, y, n_splits=5):
    """Evaluate model using K-Fold cross-validation"""
    
    rfc = RandomForestClassifier(
        n_estimators=100,
        random_state=68,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt'
    )
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=68)
    cv_scores = cross_val_score(rfc, X, y, cv=skf, scoring='accuracy')
    
    return cv_scores


if __name__ == "__main__":
    print("Loading data...")
    preprocessed = preprocess_pipeline('Data/Faults.csv')
    X = preprocessed['X']
    y = preprocessed['y']
    
    print("\nPerforming 5-Fold Cross-Validation...")
    cv_scores = evaluate_with_cross_validation(X, y, n_splits=5)
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5 Folds)")
    print("="*60)
    for i, score in enumerate(cv_scores, 1):
        print(f"Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print("-"*60)
    print(f"Average Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"Std Deviation:    {cv_scores.std():.4f} (±{cv_scores.std()*100:.2f}%)")
    print(f"Min Accuracy:     {cv_scores.min():.4f} ({cv_scores.min()*100:.2f}%)")
    print(f"Max Accuracy:     {cv_scores.max():.4f} ({cv_scores.max()*100:.2f}%)")
    print("="*60)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.barh(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'], cv_scores, color='steelblue')
    plt.axvline(cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
    plt.xlabel('Accuracy')
    plt.title('Cross-Validation Accuracy per Fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('evaluation/cv_results.png', dpi=100, bbox_inches='tight')
    print("\n✓ Visualization saved to: evaluation/cv_results.png")
