import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv("brain_stroke.csv")
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

df = load_data()
model, scaler = train_model(df)

st.title('Prediksi Stroke')

gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
age = st.slider('Usia', 0, 100, 50)
hypertension = st.selectbox('Hipertensi', ['Tidak', 'Ya'])
heart_disease = st.selectbox('Penyakit Jantung', ['Tidak', 'Ya'])
ever_married = st.selectbox('Pernah Menikah', ['Tidak', 'Ya'])
work_type = st.selectbox('Jenis Pekerjaan', ['Anak-anak', 'Swasta', 'Wiraswasta', 'Pemerintah', 'Tidak Bekerja'])
residence_type = st.selectbox('Tempat Tinggal', ['Perkotaan', 'Pedesaan'])
avg_glucose_level = st.number_input('Rata-rata Glukosa', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox('Status Merokok', ['Tidak Pernah', 'Pernah', 'Merokok', 'Tidak Diketahui'])

# Mapping input
input_data = pd.DataFrame({
    'gender': [0 if gender == 'Laki-laki' else 1],
    'age': [age],
    'hypertension': [0 if hypertension == 'Tidak' else 1],
    'heart_disease': [0 if heart_disease == 'Tidak' else 1],
    'ever_married': [0 if ever_married == 'Tidak' else 1],
    'work_type': [work_type],
    'Residence_type': [0 if residence_type == 'Perkotaan' else 1],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

input_scaled = scaler.transform(input_data)

if st.button('Prediksi'):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error('Pasien berisiko terkena stroke.')
    else:
        st.success('Pasien tidak berisiko terkena stroke.')
