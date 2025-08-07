# Loan-Eligibility-Project

This project predicts whether a loan should be approved based on applicant details using a machine learning model. It includes a Streamlit web app for real-time predictions.

**What It Does**

Takes user input like income, loan amount, credit score, and assets

Uses a trained machine learning model to predict approval

Displays result instantly through a web interface

**Tools Used**

Python

Pandas, NumPy, scikit-learn

Streamlit

Pickle (for saving the model)

**How It Works**

Data is cleaned and preprocessed

A model is trained on the dataset

The model and scaler are saved

Streamlit is used to build the user interface

The app is deployed online

**Files**

app.py — Streamlit app

loan_model.pkl — Trained model

requirements.txt — Python dependencies

**How to Run**

Clone the repo

Install dependencies with pip install -r requirements.txt

Run the app using streamlit run app.py

Note
This app uses a sample dataset and is for demonstration purposes only.
