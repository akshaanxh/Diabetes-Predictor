import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("C:/Users/HII/Desktop/MACHINE LEARNING/ML PROJECTS/DIABETES PREDICTION USING SVM/trained_model.sav", 'rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return '🟢 The person is **not diabetic**.'
    else:
        return '🔴 The person **is diabetic**.'

def main():
    st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
    st.title("🧠 Diabetes Prediction App")
    st.markdown("""
    This app predicts whether a person is **diabetic or not** using their health data.  
    Just fill in the details and hit **Predict**!
    """)

    # Input fields
    st.header("📝 Enter Patient Health Information")

    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0)

    # Prediction button
    if st.button("🔍 Predict"):
        user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        result = diabetes_prediction(user_input)
        st.subheader("🔎 Result")
        st.success(result if 'not' in result else '')
        st.error(result if 'is diabetic' in result else '')

# ✅ This part makes sure Streamlit only runs main() when executing this file
if __name__ == "__main__":
    main()
