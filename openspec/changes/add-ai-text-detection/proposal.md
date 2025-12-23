# Change: Add AI Text Detection Capability

## Why
We need to analyze text articles to determine if they are AI-generated or human-written. This is a critical capability for content verification and academic integrity.

## What Changes
- Add a new capability `ai-text-detection`
- Implement TF-IDF feature extraction using scikit-learn
- Train a Logistic Regression classifier
- Create a Streamlit web interface for text analysis
- Provide probability scores and classification results

## Impact
- Affected specs: `ai-text-detection` (new)
- Affected code: 
  - `src/model/` (new) - ML model training and prediction
  - `src/app.py` (new) - Streamlit application
  - `requirements.txt` (new) - Python dependencies
