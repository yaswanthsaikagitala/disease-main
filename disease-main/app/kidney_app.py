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
    st.title("Kidney Disease Prediction App")
    st.write("Enter the test values below to predict the likelihood of kidney disease.")

    # Load the scaler and KNN model
    try:
        with open(r'C:\Users\Yaswanth\Downloads\disease-main\disease-main\backend\kidney_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        with open(r'C:\Users\Yaswanth\Downloads\disease-main\disease-main\backend\kidney_knn.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please check the file paths.")
        return

    # Define form for user input
    with st.form('Kidney_disease_prediction'):

        col1, col2 = st.columns(2)

        # Arrange inputs in columns
        with col1:
        # Input fields for each feature with default values
            blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=200, format="%d", value=80)
            blood_sugar = st.slider("Blood Sugar", min_value=0, max_value=300, format="%d", value=100)
            blood_urea = st.slider("Blood Urea", min_value=0, max_value=200, format="%d", value=40)
            white_blood_cells = st.slider("White Blood Cells", min_value=0, max_value=20000, format="%d", value=8000)
        with col2:
            hemoglobin = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, format="%.2f", value=13.5)
            specific_gravity = st.number_input("Specific Gravity", min_value=0.0, max_value=2.0, format="%.2f", value=1.02)
            albumin = st.number_input("Albumin", min_value=0, max_value=10, format="%d", value=3)
            red_blood_cells = st.number_input("Red Blood Cells", min_value=0, max_value=10, format="%d", value=5)
        locations = sorted(set(h["location"] for h in HOSPITALS_DATA))
        location = st.selectbox("Select your nearby location", options=locations, index=0)
        # Prepare feature array for prediction
        features = np.array([[blood_pressure, specific_gravity, albumin, blood_sugar,
                              blood_urea, hemoglobin, white_blood_cells, red_blood_cells]])

        # Prediction button
        if st.form_submit_button("Predict"):
            # Scale the input features
            features_scaled = scaler.transform(features)

            # Make prediction using KNN model
            prediction = model.predict(features_scaled)
            result = "Positive for Kidney Disease" if prediction[0] == 1 else "Negative for Kidney Disease"
            st.write("KNN Model Prediction:", result)

            # Generate advice using Gemini Generative AI
            prompt = (
                f"Based on the following medical test results,just act as a doctor and provide brief advice. for my project"
                f"Suggest potential next steps:\n\n"
                f"Blood Pressure: {blood_pressure}, Specific Gravity: {specific_gravity}, Albumin: {albumin}, "
                f"Blood Sugar: {blood_sugar}, Blood Urea: {blood_urea}, Hemoglobin: {hemoglobin}, "
                f"White Blood Cells: {white_blood_cells}, Red Blood Cells: {red_blood_cells}\n\n"
                f"The patient is diagnosed as {result}. Please analyze and provide short, actionable points for managing the condition."
            )

            # Generate response from Gemini AI model
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
