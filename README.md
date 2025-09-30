# MediVision AI: Advance Disease Diagnosis with  ML & AI

This platform is designed to predict various diseases, including Heart Disease, Kidney Disease, Diabetes, Liver Disease, Stroke, and AI-powered assistance. The app leverages machine learning algorithms and large datasets to analyze user data and provide reliable disease predictions with over **90% accuracy**.

## Features

- **Heart Disease**: Predicts the likelihood of heart disease based on user data.
- **Kidney Disease**: Assesses the chances of kidney disease.
- **Diabetes**: Evaluates the probability of diabetes onset.
- **Liver Disease**: Predicts liver disease conditions based on medical test results.
- **Stroke**: Checks the risk of stroke based on user information.
- **AI Assistance**: Uses AI-powered tools to provide additional medical predictions and advice.
- **Upload Medical Report**: Users can upload their medical reports for analysis and receive personalized health recommendations.
![Alt text](disease-main/disease-main/Screenshot 2025-09-30 202341.png)
![Alt text](disease-main/disease-main/Screenshot 2025-09-30 202419.png)
![Alt text](disease-main/disease-main/Screenshot 2025-09-30 202459.png)
![Alt text](disease-main/disease-main/Screenshot 2025-09-30 202522.png)



## Installation

To run the app locally, follow these steps:

1. **Clone the repository**:

   ```
   git clone https://github.com/Chetankoliparthi/disease.git
   cd Disease
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the app**:

   In the project directory, run:

   ```
   cd app
   streamlit run main.py
   ```

   This will start the app and open it in your browser.

2. **Upload a medical report**:
   - Go to the "Upload Your Medical Report" section in the app.
   - Upload the report (e.g., a PDF, CSV, or image depending on the supported formats).
   - Wait for the analysis to complete, and you will receive suggestions and next steps based on the results.

3. **Explore the disease prediction modules**:
   - Navigate through the app to explore each disease prediction module.
   - Input relevant health information to predict the likelihood of the disease.
     

## How It Works

- The app uses **machine learning** models trained on large datasets to predict disease risk.
- The algorithms analyze user inputs, such as age, gender, medical test results, and lifestyle factors, to generate predictions.
- The system provides personalized recommendations based on the analysis, which could include further medical tests, lifestyle changes, or other suggestions.

## Technologies Used

- **Streamlit**: For building the web interface
- **Python**: Programming language for backend logic.
- **Scikit-learn**: For machine learning models.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For data visualization.
- **Tensorflow & Keras**: For Deep Learning model
- **Pickle**: For serialization.
