# 🔥 AI-Based-Wildfire-Prediction-Using-Climate-Data
This project implements a robust wildfire detection model using a hybrid approach of traditional machine learning (XGBoost) for feature selection and a deep learning-based Temporal Transformer model for classification. Additionally, it includes SHAP-based interpretability and a user-friendly CLI prediction interface.

📁 Project Structure

📦 wildfire-prediction-transformer
┣ 📜 final_wf.py               # Main training, evaluation, and prediction script

┣ 📜 Fire_dataset_cleaned.csv  # Cleaned input dataset

┣ 📜 temporal_transformer_model.pth # Trained transformer model

┣ 📜 scaler.pkl                # Saved scaler for normalization

┣ 📜 README.md                 # Project documentation

🚀 Features

Feature Selection with XGBoost

Deep Learning with Temporal Transformer (PyTorch)

SHAP for Interpretability

Robust Preprocessing using RobustScaler

Interactive Prediction Interface with CLI

📊 Dataset
The dataset used is Fire_dataset_cleaned.csv, which includes meteorological features such as:

Temperature, RH (Relative Humidity), Ws (Wind speed), Rain

Fire indices: FFMC, DMC, DC, ISI, BUI, FWI

Label: Classes (fire or not fire)

🧠 Model Architecture
🔹 XGBoost
Used for initial training to determine top 10 most important features.

🔹 Temporal Transformer
A PyTorch-based Transformer Encoder for classification using selected features.

Regularization with Dropout.

Early Stopping applied for optimal performance.

📈 Training & Evaluation
Run the main script to train the model:



python final_wf.py
The training includes:

Data preprocessing

XGBoost feature selection

Temporal Transformer model training

Evaluation and Accuracy reporting

🔍 Explainability
SHAP values are calculated to interpret XGBoost model predictions.

Generates a SHAP summary plot to visualize feature importance.

🤖 CLI Prediction Interface
After training, use the interactive command-line interface to input features and predict wildfire occurrence:

python final_wf.py
Follow the prompts to input values like Temperature, RH, Ws, etc., and the model will output:

Raw logits

Softmax probabilities

Final prediction (Fire / No Fire)

💾 Saved Files
temporal_transformer_model.pth: Trained Transformer model weights

scaler.pkl: Scaler used for consistent normalization during predictions

📦 Requirements
Install the required dependencies using:


pip install -r requirements.txt
Dependencies include:

torch

xgboost

pandas

numpy

scikit-learn

shap

termcolor

joblib

📌 Notes
Trained on top 10 features selected by XGBoost.

Transformer trained with early stopping for better generalization.

Supports gradient clipping and dropout to avoid overfitting.

🧠 Future Enhancements
Deploy as a Flask web app or Streamlit dashboard

Real-time prediction integration with weather APIs

Alert system for fire-prone areas

👩‍💻 Author

Parvathala Gayathri
