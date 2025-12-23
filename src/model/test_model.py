"""
AI Text Detection Model Testing Script

This script evaluates the trained model on the test dataset.
"""

import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from train import load_texts
from predict import AITextDetector


def prepare_test_data():
    """Load and prepare test data from files."""
    print("Loading test data...")
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Load AI-generated texts
    ai_path = os.path.join(project_root, 'data', 'test', 'ai_generated.txt')
    ai_texts = load_texts(ai_path)
    
    # Load human-written texts
    human_path = os.path.join(project_root, 'data', 'test', 'human_written.txt')
    human_texts = load_texts(human_path)
    
    # Combine texts and create labels
    X = ai_texts + human_texts
    y = [1] * len(ai_texts) + [0] * len(human_texts)  # 1 = AI, 0 = Human
    
    print(f"Loaded {len(ai_texts)} AI-generated test texts")
    print(f"Loaded {len(human_texts)} human-written test texts")
    print(f"Total test samples: {len(X)}")
    
    return X, y


def test_model():
    """Test the trained model on the test dataset."""
    print("\n" + "="*60)
    print("AI TEXT DETECTION MODEL TESTING")
    print("="*60 + "\n")
    
    # Load test data
    X_test, y_test = prepare_test_data()
    
    # Create detector
    print("\nLoading trained model...")
    detector = AITextDetector()
    
    # Make predictions
    print("Making predictions on test set...")
    predictions = []
    for text in X_test:
        result = detector.predict_text(text)
        # Convert label to numeric: AI-generated = 1, Human-written = 0
        pred = 1 if result['label'] == 'AI-generated' else 0
        predictions.append(pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # Print classification report
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, predictions, 
                                target_names=['Human-written', 'AI-generated']))
    
    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print("\nConfusion Matrix Interpretation:")
    print(f"  True Negatives (Human correctly identified): {cm[0][0]}")
    print(f"  False Positives (Human misclassified as AI): {cm[0][1]}")
    print(f"  False Negatives (AI misclassified as Human): {cm[1][0]}")
    print(f"  True Positives (AI correctly identified): {cm[1][1]}")
    
    # Show detailed predictions
    print("\n" + "="*60)
    print("DETAILED PREDICTIONS")
    print("="*60)
    
    for i, (text, true_label, pred_label) in enumerate(zip(X_test, y_test, predictions), 1):
        result = detector.predict_text(text)
        true_label_str = 'AI-generated' if true_label == 1 else 'Human-written'
        
        print(f"\nSample {i}:")
        print(f"Text: {text[:100]}...")
        print(f"True Label: {true_label_str}")
        print(f"Predicted: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        
        # Mark if prediction is correct
        if true_label == pred_label:
            print("[CORRECT]")
        else:
            print("[INCORRECT]")
        print("-"*60)
    
    print("\n" + "="*60)
    print("MODEL TESTING COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    test_model()
