"""
AI Text Detection Model Training Script

This script trains a Logistic Regression classifier using TF-IDF features
to distinguish between AI-generated and human-written text.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def load_texts(filepath):
    """Load text samples from a file, splitting by double newlines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split by double newlines to separate individual texts
    texts = [text.strip() for text in content.split('\n\n') if text.strip()]
    return texts


def prepare_training_data():
    """Load and prepare training data from files."""
    print("Loading training data...")
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Load AI-generated texts
    ai_path = os.path.join(project_root, 'data', 'train', 'ai_generated.txt')
    ai_texts = load_texts(ai_path)
    
    # Load human-written texts
    human_path = os.path.join(project_root, 'data', 'train', 'human_written.txt')
    human_texts = load_texts(human_path)
    
    # Combine texts and create labels
    X = ai_texts + human_texts
    y = [1] * len(ai_texts) + [0] * len(human_texts)  # 1 = AI, 0 = Human
    
    print(f"Loaded {len(ai_texts)} AI-generated texts")
    print(f"Loaded {len(human_texts)} human-written texts")
    print(f"Total training samples: {len(X)}")
    
    return X, y


def train_model():
    """Train the AI text detection model."""
    print("\n" + "="*60)
    print("AI TEXT DETECTION MODEL TRAINING")
    print("="*60 + "\n")
    
    # Load training data
    X_train, y_train = prepare_training_data()
    
    # Create TF-IDF vectorizer
    print("\nCreating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Limit vocabulary size
        ngram_range=(1, 2),     # Use unigrams and bigrams
        min_df=2,               # Minimum document frequency
        stop_words='english',   # Remove common English stop words
        lowercase=True
    )
    
    # Transform texts to TF-IDF features
    print("Transforming texts to TF-IDF features...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression classifier...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        C=1.0
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate on training data
    y_pred = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train, y_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    print("\nClassification Report (Training Data):")
    print(classification_report(y_train, y_pred, 
                                target_names=['Human-written', 'AI-generated']))
    
    # Save model and vectorizer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    models_dir = os.path.join(project_root, 'models')
    
    model_path = os.path.join(models_dir, 'model.pkl')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    
    print(f"\nSaving model to: {model_path}")
    joblib.dump(model, model_path)
    
    print(f"Saving vectorizer to: {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Display top features for each class
    print("\nTop 10 features for AI-generated text:")
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    top_ai_indices = np.argsort(coef)[-10:][::-1]
    for idx in top_ai_indices:
        print(f"  - {feature_names[idx]}: {coef[idx]:.4f}")
    
    print("\nTop 10 features for Human-written text:")
    top_human_indices = np.argsort(coef)[:10]
    for idx in top_human_indices:
        print(f"  - {feature_names[idx]}: {coef[idx]:.4f}")
    
    return model, vectorizer


if __name__ == "__main__":
    train_model()
