# 💳 Credit Card Fraud Detection

This project implements a **machine learning pipeline** for detecting fraudulent credit card transactions.  
It includes **data preprocessing, model training, hyperparameter tuning, and a Streamlit web app** for interactive predictions.

---

## 🚀 Features
- ✅ Data preprocessing with **StandardScaler** (feature scaling)
- ✅ Handles **class imbalance** using weighted models
- ✅ Multiple ML models: Logistic Regression, Random Forest, and XGBoost
- ✅ Hyperparameter tuning with **RandomizedSearchCV**
- ✅ Streamlit app to upload CSV files and detect fraud
- ✅ Model + Scaler persistence using **joblib**

---

## 📂 Project Structure

credit-card-fraud-detection/
│── app/ # Streamlit app
│ └── app.py # Main app script
│
│── data/ # Data & saved artifacts
│ ├── raw/creditcard.csv # Original dataset
│ ├── processed_data.pkl # Preprocessed data
│ ├── scaler.pkl # Fitted StandardScaler
│ ├── feature_columns.pkl # Feature names
│ └── fraud_model.pkl # Trained model
│
│── notebooks/ # Jupyter notebooks (optional)
│ ├── preprocessing.ipynb
│ └── fraud_model.ipynb
│
│── README.md # Project documentation
│── requirements.txt # Dependencies

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/akashrbhat1-cpu/credit-card-fraud-detection.git
cd credit-card-fraud-detection

### 2 Set up the environment

### Using Conda (recommended):

conda env create -f environment.yml
conda activate fraud_env

## Or using pip:

pip install -r requirements.txt

###🚀 Run the Web App

## From the project root directory:

streamlit run app/app.py

Then open the link shown in the terminal (usually http://localhost:8501).
📊 Model Training

    Preprocess the dataset
    Run notebooks/preprocessing.ipynb to clean and scale the data.

    Train models
    Run notebooks/fraud_model.ipynb to train Logistic Regression, Random Forest, and XGBoost.
    The best model (fraud_model.pkl) will be saved in the data/ folder.

🛠️ Features

    Handles imbalanced dataset using scaling and class_weight.

    Multiple models trained and compared.

    Saves best model, scaler, and feature columns for reproducibility.

    Streamlit web app to upload transactions and detect fraud.

📌 Requirements

    Python 3.9+

    Libraries: Pandas, Scikit-learn, XGBoost, Matplotlib, Streamlit, Joblib

(See requirements.txt or environment.yml for full list)
📜 License

This project is licensed under the MIT License.