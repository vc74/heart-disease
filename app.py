import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0;
    }
    .subtitle {
        color: #f0f0f0;
        text-align: center;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        border: none;
        border-radius: 30px;
        padding: 15px 60px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .result-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 30px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .healthy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .warning {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    div[data-testid="stHorizontalBlock"] {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    label {
        font-weight: 600;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Header
st.markdown("<h1>❤️ Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered health assessment tool • Enter your details below</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input form
st.markdown("---")

# Personal Information Section
st.markdown("### 👤 Personal Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age (years)', min_value=1, max_value=120, value=30, help="Enter your current age in years")
    
with col2:
    sex = st.selectbox('Sex', options=['Female', 'Male'], help="Select your biological sex")

fbs = st.number_input('Fasting Blood Sugar (mg/dl)', min_value=0, max_value=300, value=100, 
                      help="Blood sugar after 8+ hours fasting. Normal: <100, Prediabetes: 100-125, Diabetes: >126")

st.info("ℹ️ **Fasting Blood Sugar Guide:** Normal range is below 100 mg/dl. If you haven't had this test, use an estimated value or leave at default.")

st.markdown("---")

# Medical Measurements Section
st.markdown("### 🩺 Medical Measurements")

col1, col2 = st.columns(2)

with col1:
    restingbp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120,
                                help="Your blood pressure while at rest (systolic value)")
    cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=0, max_value=600, value=200,
                                  help="Total cholesterol from blood test")

with col2:
    maxhr = st.number_input('Max Heart Rate Achieved', min_value=50, max_value=250, value=150,
                           help="Maximum heart rate during exercise. Typical max = 220 - your age")
    oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=-5.0, max_value=10.0, value=0.0, step=0.1,
                             help="ST segment change on ECG during exercise")

st.info("ℹ️ **Blood Pressure:** Normal <120, Elevated 120-129, High ≥130 | **Cholesterol:** Desirable <200, Borderline 200-239, High ≥240")

st.markdown("---")

# Clinical Tests Section
st.markdown("### 📊 Clinical Test Results")

chest_pain = st.selectbox(
    'Chest Pain Type', 
    options=['ASY', 'ATA', 'NAP', 'TA'], 
    format_func=lambda x: {
        'ASY': 'Asymptomatic (No chest pain)', 
        'ATA': 'Atypical Angina (Unusual chest discomfort)', 
        'NAP': 'Non-Anginal Pain (Chest pain not heart-related)', 
        'TA': 'Typical Angina (Classic heart-related chest pain)'
    }[x],
    help="Select the type of chest pain you experience"
)

with st.expander("ℹ️ What do these chest pain types mean?"):
    st.write("""
    - **Typical Angina (TA):** Classic heart-related chest pain with pressure, tightness, or squeezing sensation
    - **Atypical Angina (ATA):** Chest discomfort that doesn't follow the classic pattern
    - **Non-Anginal Pain (NAP):** Chest pain unlikely to be related to your heart
    - **Asymptomatic (ASY):** No chest pain symptoms
    """)

col1, col2 = st.columns(2)

with col1:
    resting_ecg = st.selectbox(
        'Resting ECG Results', 
        options=['Normal', 'ST', 'LVH'], 
        format_func=lambda x: {
            'Normal': 'Normal', 
            'ST': 'ST-T Wave Abnormality', 
            'LVH': 'Left Ventricular Hypertrophy (Enlarged heart)'
        }[x],
        help="Results from your electrocardiogram at rest"
    )
    
    exercise_angina = st.selectbox(
        'Exercise Induced Angina', 
        options=['No', 'Yes'],
        help="Do you experience chest pain during physical activity?"
    )

with col2:
    st_slope = st.selectbox(
        'ST Slope During Peak Exercise', 
        options=['Up', 'Flat', 'Down'], 
        format_func=lambda x: {
            'Up': 'Upsloping (Typically healthy)', 
            'Flat': 'Flat (May indicate concern)', 
            'Down': 'Downsloping (May indicate concern)'
        }[x],
        help="Shape of ST segment on ECG during peak exercise"
    )

with st.expander("ℹ️ What are ECG results?"):
    st.write("""
    - **Normal:** Healthy heart electrical activity with no abnormalities
    - **ST-T Wave Abnormality:** Indicates possible heart strain or reduced blood flow
    - **Left Ventricular Hypertrophy:** Thickened heart wall, often caused by high blood pressure
    - **ST Slope:** Upsloping is generally healthy; Flat or Downsloping may suggest heart disease
    """)

st.markdown("---")

st.markdown("<br>", unsafe_allow_html=True)

# Predict button (centered)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button('🔍 Predict Heart Disease')

# Prediction
if predict_button:
    with st.spinner('Analyzing your health data...'):
        # Create input data with one-hot encoding
        fastingbs = 1 if fbs > 120 else 0
        
        input_dict = {
            'Age': age,
            'RestingBP': restingbp,
            'Cholesterol': cholesterol,
            'FastingBS': fastingbs,
            'MaxHR': maxhr,
            'Oldpeak': oldpeak,
            'Sex_M': 1 if sex == 'Male' else 0,
            'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
            'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
            'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
            'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
            'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
            'ExerciseAngina_Y': 1 if exercise_angina == 'Yes' else 0,
            'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
            'ST_Slope_Up': 1 if st_slope == 'Up' else 0
        }
        
        input_data = pd.DataFrame([input_dict])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        
        if prediction[0] == 1:
            st.markdown("""
                <div class='result-box warning'>
                    ⚠️ Heart Disease Detected<br>
                    <span style='font-size: 1rem; font-weight: normal;'>
                    Please consult with a healthcare professional for proper diagnosis and treatment.
                    </span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-box healthy'>
                    ✅ No Heart Disease Detected<br>
                    <span style='font-size: 1rem; font-weight: normal;'>
                    Your heart appears healthy! Continue maintaining a healthy lifestyle.
                    </span>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; opacity: 0.8;'>⚕️ This is an AI prediction tool and should not replace professional medical advice.</p>", unsafe_allow_html=True)
