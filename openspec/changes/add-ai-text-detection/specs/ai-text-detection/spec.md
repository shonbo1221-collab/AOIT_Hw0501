## ADDED Requirements

### Requirement: Text Classification
The system SHALL classify input text as either AI-generated or human-written using TF-IDF feature extraction and Logistic Regression.

#### Scenario: Successful Classification
- **WHEN** a user submits a text article through the Streamlit interface
- **THEN** the system returns a classification label (AI-generated or Human-written) with a probability score

#### Scenario: Confidence Score Display
- **WHEN** classification is performed
- **THEN** the system displays the confidence score as a percentage (0-100%)

### Requirement: Model Training
The system SHALL train a Logistic Regression model using TF-IDF features extracted from labeled training data.

#### Scenario: Model Persistence
- **WHEN** the model is trained
- **THEN** it is saved to disk for reuse in predictions

#### Scenario: Feature Extraction
- **WHEN** text is processed
- **THEN** TF-IDF vectorization is applied with consistent parameters (max_features, ngram_range)

### Requirement: Web Interface
The system SHALL provide a Streamlit-based web interface for text analysis.

#### Scenario: Text Input
- **WHEN** a user accesses the application
- **THEN** they can input or paste text into a text area

#### Scenario: Real-time Prediction
- **WHEN** a user clicks the "Analyze" button
- **THEN** the prediction is displayed within 2 seconds

#### Scenario: Result Visualization
- **WHEN** prediction results are shown
- **THEN** the interface displays:
  - Classification label
  - Confidence percentage
  - Visual indicator (color-coded or progress bar)
