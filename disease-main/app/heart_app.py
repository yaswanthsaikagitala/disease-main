import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
# Configure Google Generative AI
# Get API key from environment variable
#get the key from the environment variable
api_key = os.getenv("GENAI_API_KEY")

# Use the API key
genai.configure(api_key=api_key)
# Define the main display function for the Heart Disease Prediction App
hospitals_df = pd.read_csv("hospitals_india.csv")
HOSPITALS_DATA = hospitals_df.to_dict("records")
for hospital in HOSPITALS_DATA:
    hospital["specialties"] = hospital["specialties"].split(":")

# Function to get hospitals specializing in Liver
def get_nearby_hospitals(selected_location):
    hospitals = [h for h in HOSPITALS_DATA if h["location"] == selected_location and "Liver" in h["specialties"]]
    if not hospitals:
        return ["No liver-specialized hospitals found for this location."]
    return [f"- **{h['name']}**: {h['address']} (Specialties: {', '.join(h['specialties'])})" for h in hospitals]
def display():
    st.title("Heart Disease Prediction App")
    st.write("Enter the details below to predict the likelihood of heart disease.")

    # Load the scaler and KNN model
    try:
        with open(r"C:\Users\Yaswanth\Downloads\disease-main\disease-main\backend\heart_scaler.pkl", 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        with open(r"C:\Users\Yaswanth\Downloads\disease-main\disease-main\backend\heart_knn.pkl", 'rb') as knn_file:
            knn_model = pickle.load(knn_file)
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please check the file paths.")
        return

    # Define form for user input
    with st.form("Heart_disease_form"):

        col1, col2 = st.columns(2)

        # Arrange inputs in columns
        with col1:
        # Input fields for heart disease prediction
            age = st.slider("Age", min_value=0, max_value=120, step=1, value=50)
            resting_bp = st.slider("Resting Blood Pressure", min_value=0, max_value=300, value=120, step=1)
            cholesterol = st.slider("Cholesterol", min_value=0, max_value=600, value=200, step=5)
            max_hr = st.slider("Max Heart Rate Achieved", min_value=0, max_value=300, value=150, step=5)
            oldpeak = st.slider("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"], index=1)
            resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LHV"], index=0)
            sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
            chest_pain_type = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"], index=0)
            exercise_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"], index=1)
            st_slope = st.selectbox("ST Slope", options=["Flat", "Up", "Down"], index=0)
        locations = sorted(set(h["location"] for h in HOSPITALS_DATA))
        location = st.selectbox("Select your nearby location", options=locations, index=0)

        # Map categorical inputs to numerical values
        sex = 0 if sex == "Male" else 1
        chest_pain_type = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain_type]
        fasting_bs = 1 if fasting_bs == "Yes" else 0
        resting_ecg = {"Normal": 0, "ST": 1, "LHV": 2}[resting_ecg]
        exercise_angina = 1 if exercise_angina == "Yes" else 0
        st_slope = {"Flat": 0, "Up": 1, "Down": 2}[st_slope]

        # Prepare the input data for prediction
        input_data = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
                                resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Prediction button
        if st.form_submit_button("Predict"):
            # KNN prediction
            knn_prediction = knn_model.predict(scaled_data)
            knn_result = "Heart Disease" if knn_prediction[0] == 1 else "No Heart Disease"
            st.write(f"KNN Model Prediction: {knn_result}")

            # Prepare prompt for Generative AI model
            prompt = (
                f"Based on the following medical details,just act as a doctor and provide brief advice. for my project"
                f"Provide the best advice and a possible diagnosis:\n\n"
                f"Age: {age}, Sex: {'Male' if sex == 0 else 'Female'}, Chest Pain Type: {chest_pain_type}, "
                f"Resting Blood Pressure: {resting_bp}, Cholesterol: {cholesterol}, "
                f"Fasting Blood Sugar > 120 mg/dl: {'Yes' if fasting_bs == 1 else 'No'}, "
                f"Resting ECG: {resting_ecg}, Max Heart Rate Achieved: {max_hr}, "
                f"Exercise Induced Angina: {'Yes' if exercise_angina == 1 else 'No'}, Oldpeak: {oldpeak}, "
                f"ST Slope: {st_slope}\n\n"
                f"I have been diagnosed with {knn_result}. Please analyze and suggest potential next steps "
                f"for managing the condition, and make the response concise and in bullet points."
            )

            # Generate response from Generative AI model
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if response:
                    st.write("**Medical Advice:**")
                    st.write(response.text)
                else:
                    st.write("No response generated. Check your input.")
            except Exception as e:
                st.error(f"An error occurred during AI response generation: {e}")
            st.write(f"**Recommended Hospitals in {location} for Heart Disease:**")
            hospitals = get_nearby_hospitals(location)
            for hospital in hospitals:
                st.markdown(hospital)

if __name__ == '__main__':
    display()
