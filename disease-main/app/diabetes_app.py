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
# Define the main display function
def display():
    st.title("Diabetes Prediction App")

    # Load required resources
    try:
        # Load the scaler
        with open(r"C:\Users\Yaswanth\Downloads\disease-main\disease-main\backend\diabetes_scaler.pkl", 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Load the KNN model
        with open(r"C:\Users\Yaswanth\Downloads\disease-main\disease-main\backend\diabetes_knn.pkl", 'rb') as knn_file:
            knn_model = pickle.load(knn_file)
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please check the file paths.")
        return

    # Define form for user input
    with st.form("Diabetes_disease_form"):

        col1, col2 = st.columns(2)

        # Arrange inputs in columns
        with col1:
            BMI = st.slider("BMI", min_value=10, max_value=50, value=25)
            Age = st.slider("Age", min_value=0, max_value=120, value=30)
            Glucose = st.slider("Glucose", min_value=0, max_value=200, value=100)
            BloodPressure = st.slider("BloodPressure", min_value=40, max_value=200, value=70)
        with col2:
            Insulin = st.number_input("Insulin", min_value=0, max_value=600, value=100)
            DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.5)
            Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=120, step=1, value=1)
            SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=300, value=20)
        locations = sorted(set(h["location"] for h in HOSPITALS_DATA))
        location = st.selectbox("Select your nearby location", options=locations, index=0)

        # Prepare the input data for prediction
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        scaled_data = scaler.transform(input_data)

        # Form submit button
        if st.form_submit_button("Predict"):
            # Make prediction using the KNN model
            knn_prediction = knn_model.predict(scaled_data)
            knn_result = "You have Diabetes" if knn_prediction == 1 else "You don't have Diabetes"
            st.write(f"KNN Model Prediction: {knn_result}")

            # Prepare prompt for Generative AI model
            prompt = (
                f"Based on the following medical details,just act as a doctor and provide brief advice. for my project"
                f"Provide the best advice and a possible diagnosis:\n\n"
                f"Pregnancies: {Pregnancies}, Glucose: {Glucose}, Blood Pressure: {BloodPressure}, "
                f"Skin Thickness: {SkinThickness}, Insulin: {Insulin}, BMI: {BMI}, "
                f"Diabetes Pedigree Function: {DiabetesPedigreeFunction}, Age: {Age}\n\n"
                f"I have been diagnosed with {knn_result}. Please analyze and suggest potential next steps "
                f"for managing the condition, and make the response concise and in bullet points."
            )

            # Generate response from Generative AI model
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if response:
                    st.write("**Suggestion:**")
                    st.write(response.text)
                else:
                    st.write("No response generated. Check your input.")
            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")

            st.write(f"**Recommended Hospitals in {location} for Heart Disease:**")
            hospitals = get_nearby_hospitals(location)
            for hospital in hospitals:
                st.markdown(hospital)

if __name__ == '__main__':
    display()
