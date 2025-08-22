# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.

## Dataset
The dataset is based on anonymized credit card transactions. Features `V1` to `V28` are the result of PCA transformation, with additional features `Time` and `Amount`.

## Approach
- **Preprocessing**: Handled class imbalance using **SMOTE**.
- **Models Tried**: Logistic Regression and Random Forest.
- **Model Selection**: Random Forest chosen based on best **PR Score** and **ROC AUC**.

## Saved Model
The final trained model (Random Forest) along with the best threshold and selected features is saved using **pickle**.

## Streamlit App
A simple **Streamlit UI** is provided where users can upload a CSV file containing transactions (`Time, V1, V2, ..., V28, Amount`) and get predictions on whether they are fraudulent.


