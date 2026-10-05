import streamlit as st
import pandas as pd
import joblib 

model = joblib.load('logistic_model.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Logistic Regression Model Prediction👨‍💻")
st.markdown('Provide the following details')

age = st.slider('Age', 20, 100, 50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['0', '1', '2', '3'])
trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
chol = st.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['0', '1'])
restecg = st.selectbox('Resting Electrocardiographic Results', ['0', '1', '2'])
thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exang = st.selectbox('Exercise Induced Angina', ['0', '1'])
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.0, 1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['0', '1', '2'])
ca = st.selectbox('Number of Major Vessels (0-3) Colored by Fluoroscopy', ['0', '1', '2', '3'])
thal = st.selectbox('Thalassemia', ['0', '1', '2', '3'])


if st.button('Predict'):
    raw_input = {
        
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'cp': int(cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': int(fbs),
        'restecg': int(restecg),
        'thalach': thalach,
        'exang': int(exang),
        'oldpeak': oldpeak,
        'slope': int(slope),
        'ca': int(ca),
        'thal': int(thal)
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("HIGH RISK OF HEART DISEASE🏥")
    else:
        st.success("LOW RISK OF HEART DISEASE💪")

        