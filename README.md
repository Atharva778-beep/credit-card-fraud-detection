# Credit Card Fraud Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atharva-cc-fraud-detector.streamlit.app/)

A machine learning web application that predicts whether a credit card transaction is fraudulent or legitimate using a trained classification model and an interactive Streamlit interface.

## Live Demo

https://atharva-cc-fraud-detector.streamlit.app/

## Overview

This project detects fraudulent credit card transactions using a trained machine learning model. The model is deployed through Streamlit so users can enter transaction details and get predictions in real time.

## Features

- Predicts whether a transaction is fraudulent or legitimate.
- Simple and interactive Streamlit user interface.
- Uses a pre-trained machine learning model for inference.
- Deployed online using Streamlit Community Cloud.
- Suitable for portfolio and machine learning project showcase.

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Project Structure

```bash
credit_card_fraud_detection/
├── fraud_detector.py
├── requirements.txt
├── output/
│   ├── fraud_model.pkl
│   └── feature_names.pkl
├── README.md
```

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
python -m streamlit run fraud_detector.py
```

The model is already trained and saved in:

- `output/fraud_model.pkl`
- `output/feature_names.pkl`

## Deployment

This app is deployed on Streamlit Community Cloud.

Live App:  
https://atharva-cc-fraud-detector.streamlit.app/

## Author

**Atharva Sawant**  
GitHub: https://github.com/Atharva778-beep