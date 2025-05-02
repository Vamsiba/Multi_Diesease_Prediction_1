import streamlit as st
import pickle
import os
from streamlit_option_menu import option_menu

# Page settings
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="👨‍⚕️🏥")

# Load Models
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes.pkl', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart.pkl', 'rb'))
kidney_disease_model = pickle.load(open(f'{working_dir}/saved_models/kidney.pkl', 'rb'))

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction",
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Kidney Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )


if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction Using Machine Learning")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0.0, step=1.0)
    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0.0)
    with col3:
        BloodPressure = st.number_input("Blood Pressure Value", min_value=0.0)
    with col1:
        SkinThickness = st.number_input("Skin Thickness Value", min_value=0.0)
    with col2:
        Insulin = st.number_input("Insulin Value", min_value=0.0)
    with col3:
        BMI = st.number_input("BMI Value", min_value=0.0)
    with col1:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function Value", min_value=0.0)
    with col2:
        Age = st.number_input("Age", min_value=0.0, step=1.0)

    diabetes_result = ""

    if st.button("Diabetes Test Result"):
        user_input = [
            Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
            DiabetesPedigreeFunction, Age
        ]

        prediction = diabetes_model.predict([user_input])

        if prediction[0] == 1:
            diabetes_result = "The person has diabetes."
        else:
            diabetes_result = "The person does not have diabetes."

        st.success(diabetes_result)

# -------------------- Heart Disease Prediction Page --------------------
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction Using Machine Learning")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0.0, step=1.0)
    with col2:
        sex = st.number_input("Sex (1=Male, 0=Female)", min_value=0, max_value=1, step=1)
    with col3:
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
    with col1:
        trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0.0)
    with col3:
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", min_value=0, max_value=1, step=1)
    with col1:
        restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2, step=1)
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0.0)
    with col3:
        exang = st.number_input("Exercise Induced Angina (1 = yes; 0 = no)", min_value=0, max_value=1, step=1)
    with col1:
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, format="%.2f")
    with col2:
        slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, step=1)
    with col3:
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, step=1)
    with col1:
        thal = st.number_input("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", min_value=1, max_value=3, step=1)

    heart_disease_result = ""

    if st.button("Heart Disease Test Result"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        prediction = heart_disease_model.predict([user_input])

        if prediction[0] == 1:
            heart_disease_result = "The person has heart disease."
        else:
            heart_disease_result = "The person does not have heart disease."

        st.success(heart_disease_result)

# -------------------- Kidney Disease Prediction Page --------------------

            # -------------------- Kidney Disease Prediction Page --------------------
# -------------------- Kidney Disease Prediction Page --------------------
# -------------------- Kidney Disease Prediction Page --------------------
if selected == 'Kidney Disease Prediction':
    st.title("Kidney Disease Prediction Using Machine Learning")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.number_input('Age', min_value=0.0, step=1.0)
    with col2:
        blood_pressure = st.number_input('Blood Pressure', min_value=0.0)
    with col3:
        specific_gravity = st.number_input('Specific Gravity', min_value=0.0, format="%.2f")
    with col4:
        albumin = st.number_input('Albumin', min_value=0.0)
    with col5:
        sugar = st.number_input('Sugar', min_value=0.0)

    with col1:
        red_blood_cells = st.number_input('Red Blood Cells (1=normal, 0=abnormal)', min_value=0, max_value=1, step=1)
    with col2:
        pus_cell = st.number_input('Pus Cell (1=normal, 0=abnormal)', min_value=0, max_value=1, step=1)
    with col3:
        pus_cell_clumps = st.number_input('Pus Cell Clumps (1=present, 0=not present)', min_value=0, max_value=1, step=1)
    with col4:
        bacteria = st.number_input('Bacteria (1=present, 0=not present)', min_value=0, max_value=1, step=1)
    with col5:
        blood_glucose_random = st.number_input('Blood Glucose Random', min_value=0.0)

    with col1:
        blood_urea = st.number_input('Blood Urea', min_value=0.0)
    with col2:
        serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0)
    with col3:
        sodium = st.number_input('Sodium', min_value=0.0)
    with col4:
        potassium = st.number_input('Potassium', min_value=0.0)
    with col5:
        haemoglobin = st.number_input('Haemoglobin', min_value=0.0)

    with col1:
        packed_cell_volume = st.number_input('Packed Cell Volume', min_value=0.0)
    with col2:
        white_blood_cell_count = st.number_input('White Blood Cell Count', min_value=0.0)
    with col3:
        red_blood_cell_count = st.number_input('Red Blood Cell Count', min_value=0.0)
    with col4:
        hypertension = st.number_input('Hypertension (1=yes, 0=no)', min_value=0, max_value=1, step=1)
    with col5:
        diabetes_mellitus = st.number_input('Diabetes Mellitus (1=yes, 0=no)', min_value=0, max_value=1, step=1)

    with col1:
        coronary_artery_disease = st.number_input('Coronary Artery Disease (1=yes, 0=no)', min_value=0, max_value=1, step=1)
    with col2:
        appetite = st.number_input('Appetite (1=good, 0=poor)', min_value=0, max_value=1, step=1)
    with col3:
        peda_edema = st.number_input('Peda Edema (1=yes, 0=no)', min_value=0, max_value=1, step=1)
    with col4:
        anemia = st.number_input('Anemia (1=yes, 0=no)', min_value=0, max_value=1, step=1)
    with col5:
        class_value = st.number_input('Class (1=kidney disease, 0=no disease)', min_value=0, max_value=1, step=1)

    kidney_diagnosis = ""

    if st.button("Kidney's Test Result"):
        # Now 25 inputs in the same order the model expects
        user_input = [
            age, blood_pressure, specific_gravity, albumin, sugar,
            red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
            blood_glucose_random, blood_urea, serum_creatinine, sodium,
            potassium, haemoglobin, packed_cell_volume, white_blood_cell_count,
            red_blood_cell_count, hypertension, diabetes_mellitus,
            coronary_artery_disease, appetite, peda_edema, anemia,
            class_value
        ]

        prediction = kidney_disease_model.predict([user_input])

        if prediction[0] == 1:
            kidney_diagnosis = "The person has kidney disease."
        else:
            kidney_diagnosis = "The person does not have kidney disease."

        st.success(kidney_diagnosis)
