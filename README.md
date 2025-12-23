# Taiwan Weather Data Pipeline & AI Text Detection System

This project contains two main capabilities:
1. **Weather Data Pipeline** - Downloads and visualizes Taiwan weather data from CWA API
2. **AI Text Detection** - Classifies text as AI-generated or human-written

## 🌐 Live Demo

**Try the AI Text Detection System online:**  
🔗 [https://aoithw0501-nappyp3vd7s2iqtocjzpwbl.streamlit.app/](https://aoithw0501-nappyp3vd7s2iqtocjzpwbl.streamlit.app/)

No installation required - just visit the link and start analyzing text!

## 🚀 Quick Start

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### AI Text Detection System

#### 1. Train the Model
```bash
python src/model/train.py
```

This will:
- Load training data from `data/train/`
- Train a TF-IDF + Logistic Regression classifier
- Save the model to `models/` directory
- Display training accuracy and top features

#### 2. Test the Model (Optional)
```bash
python src/model/test_model.py
```

This will evaluate the model on test data and show detailed predictions.

#### 3. Run the Web Application
```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Hw05/
├── data/                          # Training and test datasets
│   ├── train/
│   │   ├── ai_generated.txt      # AI-generated training samples
│   │   └── human_written.txt     # Human-written training samples
│   └── test/
│       ├── ai_generated.txt      # AI-generated test samples
│       └── human_written.txt     # Human-written test samples
├── models/                        # Trained model files
│   ├── model.pkl                 # Logistic Regression model
│   └── vectorizer.pkl            # TF-IDF vectorizer
├── src/
│   ├── model/
│   │   ├── train.py              # Model training script
│   │   ├── predict.py            # Prediction module
│   │   └── test_model.py         # Model testing script
│   └── app.py                    # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🤖 AI Text Detection Features

### How It Works
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification**: Logistic Regression
- **Vocabulary**: 5000 most important features
- **N-grams**: Unigrams and bigrams

### Web Interface Features
- 📝 Text input area for analysis
- 🎯 Probability breakdown (AI vs Human)
- 📊 Confidence meter with color coding
- 📚 Sample texts for quick testing
- 📈 Text statistics (word count, character count)
- 💡 Confidence level interpretation

### Usage Tips
- Enter at least 5 words for accurate results
- Longer texts generally produce better predictions
- Confidence scores indicate prediction certainty:
  - ✅ **High (80%+)**: Very reliable
  - ⚠️ **Medium (60-80%)**: Moderately reliable
  - ❌ **Low (<60%)**: Less reliable

## 🌤️ Weather Data Pipeline

(Add your weather pipeline documentation here)

## 📊 Model Performance

The model is trained on sample datasets with the following characteristics:
- **Training Data**: 15 AI-generated + 15 human-written texts
- **Test Data**: 5 AI-generated + 5 human-written texts
- **Features**: TF-IDF with max 5000 features, unigrams + bigrams

> **Note**: This is a demonstration system. For production use, expand the dataset with more diverse examples.

## 🛠️ Tech Stack

- **Python 3.x**
- **scikit-learn** - Machine learning
- **Streamlit** - Web interface
- **pandas** - Data processing
- **numpy** - Numerical operations
- **joblib** - Model serialization

## 📝 Example Usage

### Command Line Prediction
```python
from src.model.predict import AITextDetector

detector = AITextDetector()
result = detector.predict_text("Your text here...")

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

### Web Interface
1. Launch the app: `streamlit run src/app.py`
2. Enter or paste text in the text area
3. Click "Analyze Text"
4. View results with confidence scores and probabilities

## 🔍 Testing

Run the test suite to evaluate model performance:
```bash
python src/model/test_model.py
```

This will show:
- Test accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Detailed predictions for each test sample

## 📄 License

This project is for educational purposes.

## 👥 Contributors

- Your Name

## 🙏 Acknowledgments

- Central Weather Administration (CWA) for weather data API
- scikit-learn for machine learning tools
- Streamlit for the web framework
