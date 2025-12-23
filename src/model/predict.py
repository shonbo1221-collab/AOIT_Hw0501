"""
AI Text Detection Prediction Module

This module provides functions to load the trained model and make predictions
on new text samples.
"""

import os
import joblib
import numpy as np


class AITextDetector:
    """AI Text Detection classifier."""
    
    def __init__(self):
        """Initialize the detector by loading the trained model and vectorizer."""
        self.model = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer from disk."""
        # Get the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        models_dir = os.path.join(project_root, 'models')
        
        model_path = os.path.join(models_dir, 'model.pkl')
        vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "Model files not found. Please run train.py first to train the model."
            )
        
        # Load model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
    
    def predict_text(self, text):
        """
        Predict whether the given text is AI-generated or human-written.
        
        Args:
            text (str): The text to classify
            
        Returns:
            dict: A dictionary containing:
                - 'label': 'AI-generated' or 'Human-written'
                - 'confidence': Confidence score (0-100%)
                - 'ai_probability': Probability of being AI-generated (0-1)
                - 'human_probability': Probability of being human-written (0-1)
        """
        # Handle edge cases
        if not text or len(text.strip()) == 0:
            return {
                'label': 'Unknown',
                'confidence': 0.0,
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'error': 'Empty text provided'
            }
        
        if len(text.split()) < 5:
            return {
                'label': 'Unknown',
                'confidence': 0.0,
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'error': 'Text too short (minimum 5 words required)'
            }
        
        # Transform text to TF-IDF features
        text_tfidf = self.vectorizer.transform([text])
        
        # Get prediction and probabilities
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Extract probabilities
        human_prob = probabilities[0]  # Class 0 = Human
        ai_prob = probabilities[1]      # Class 1 = AI
        
        # Determine label and confidence
        if prediction == 1:
            label = 'AI-generated'
            confidence = ai_prob * 100
        else:
            label = 'Human-written'
            confidence = human_prob * 100
        
        return {
            'label': label,
            'confidence': confidence,
            'ai_probability': ai_prob,
            'human_probability': human_prob
        }
    
    def predict_batch(self, texts):
        """
        Predict multiple texts at once.
        
        Args:
            texts (list): List of text strings to classify
            
        Returns:
            list: List of prediction dictionaries
        """
        return [self.predict_text(text) for text in texts]


def predict_text(text):
    """
    Convenience function to predict a single text.
    
    Args:
        text (str): The text to classify
        
    Returns:
        dict: Prediction results
    """
    detector = AITextDetector()
    return detector.predict_text(text)


if __name__ == "__main__":
    # Test the prediction module
    print("AI Text Detection - Prediction Module Test\n")
    print("="*60)
    
    # Create detector
    detector = AITextDetector()
    
    # Test samples
    test_samples = [
        "Artificial intelligence has revolutionized numerous industries in recent years. The technology enables machines to perform tasks that traditionally required human intelligence.",
        "So I was walking to the coffee shop this morning, right? And this guy literally bumps into me while staring at his phone. Doesn't even say sorry!"
    ]
    
    for i, text in enumerate(test_samples, 1):
        print(f"\nTest Sample {i}:")
        print(f"Text: {text[:100]}...")
        
        result = detector.predict_text(text)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"AI Probability: {result['ai_probability']:.4f}")
        print(f"Human Probability: {result['human_probability']:.4f}")
        print("-"*60)
