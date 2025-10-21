import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# ---------------------------
# PAGE SETUP
# ---------------------------
st.write("""
# 🩺 Diabetes Prediction App
This app predicts whether a person is **likely to have diabetes** based on health information.
""")

st.sidebar.header('User Input Parameters')

# ---------------------------
# USER INPUT FUNCTION
# ---------------------------
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
    age = st.sidebar.slider('Age', 1, 100, 40)
    hypertension = st.sidebar.selectbox('Hypertension', ('No', 'Yes'))
    heart_disease = st.sidebar.selectbox('Heart Disease', ('No', 'Yes'))
    smoking_history = st.sidebar.selectbox(
        'Smoking History', 
        ('never', 'former', 'current', 'ever', 'not current', 'No Info')
    )
    bmi = st.sidebar.slider('BMI', 10.0, 60.0, 25.0)
    HbA1c_level = st.sidebar.slider('HbA1c Level', 3.0, 15.0, 5.5)
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 50, 300, 120)
    
    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# ---------------------------
# LOAD DATASET AND TRAIN MODEL
# ---------------------------
data_path = r"diabetes_prediction_dataset.csv"

try:
    dataset = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("Dataset not found. Please check the path.")
    st.stop()

# Clean column names (remove trailing spaces, etc.)
dataset.columns = dataset.columns.str.strip()

# Ensure consistent text casing
dataset['gender'] = dataset['gender'].str.title()
dataset['smoking_history'] = dataset['smoking_history'].str.lower()
df['gender'] = df['gender'].str.title()
df['smoking_history'] = df['smoking_history'].str.lower()

# Convert Yes/No to numeric before training
for col in ['hypertension', 'heart_disease']:
    dataset[col] = dataset[col].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    df[col] = df[col].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

# Encode categorical columns
categorical_cols = ['gender', 'smoking_history']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    df[col] = le.transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = dataset.drop('diabetes', axis=1)
y = dataset['diabetes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# ---------------------------
# MAKE PREDICTIONS
# ---------------------------
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# ---------------------------
# DISPLAY RESULTS
# ---------------------------
st.subheader('Prediction')
st.write('**Diabetic** 🩸' if prediction[0] == 1 else '**Non-Diabetic** ✅')

st.subheader('Prediction Probability')
st.write(f"Non-Diabetic: {prediction_proba[0][0]:.2f} | Diabetic: {prediction_proba[0][1]:.2f}")

