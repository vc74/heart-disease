# ❤️ Heart Disease Prediction

AI-powered web application for predicting heart disease using machine learning.

## Features

- Beautiful, user-friendly Streamlit interface
- KNN-based prediction model
- Detailed input explanations for medical parameters
- Real-time predictions with visual feedback

## Files

- `app.py` - Streamlit web application
- `KNN_heart.pkl` - Trained KNN model
- `scaler.pkl` - Feature scaler
- `columns.pkl` - Column names for the model
- `heart_analysis.ipynb` - Model training notebook
- `heart.csv` - Dataset

## Installation

```bash
pip install streamlit pandas joblib scikit-learn
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Model Details

- Algorithm: K-Nearest Neighbors (KNN)
- Features: 15 medical and demographic features
- Data preprocessing: StandardScaler normalization
- One-hot encoding for categorical variables

## Disclaimer

This is an AI prediction tool and should not replace professional medical advice. Always consult with healthcare professionals for proper diagnosis and treatment.
