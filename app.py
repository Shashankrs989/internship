import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the models
logreg_model = joblib.load('logreg_model.pkl')
knn_model = joblib.load('knn_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Define feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Create user input fields
st.title("Heart Disease Prediction App")
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 45)
    sex = st.sidebar.selectbox('Sex', [0, 1])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 0)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol', 126, 564, 199)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.sidebar.slider('Resting Electrocardiographic Results (0-2)', 0, 2, 0)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 170)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 0.0)
    slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
    thal = st.sidebar.slider('Thalassemia (0-3)', 0, 3, 1)

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Load the scaler and transform the input data
scaler = joblib.load('scaler.pkl')  # Ensure the scaler is saved during model training
input_scaled = scaler.transform(input_df)

# Model prediction
model_choice = st.selectbox('Choose Model', ('Logistic Regression', 'KNN', 'Random Forest'))
if st.button('Predict'):
    if model_choice == 'Logistic Regression':
        prediction = logreg_model.predict(input_scaled)
    elif model_choice == 'KNN':
        prediction = knn_model.predict(input_scaled)
    else:
        prediction = rf_model.predict(input_scaled)
    
    result = 'heart disease' if prediction == 1 else 'no heart disease'
    st.subheader(f'The model predicts that this patient has {result}.')

# Display model accuracies
st.subheader('Model Accuracy')
st.write(f"Logistic Regression Accuracy: {logreg_model.best_score_:.2f}")
st.write(f"KNN Accuracy: {knn_model.best_score_:.2f}")
st.write(f"Random Forest Accuracy: {rf_model.best_score_:.2f}")
