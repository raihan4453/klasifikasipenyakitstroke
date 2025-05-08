import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Pastikan path ke scaler benar
scaler_path = 'scaler.sav'

# Cek apakah file scaler ada
if not os.path.exists(scaler_path):
    st.error(f"File {scaler_path} tidak ditemukan. Pastikan file scaler telah disimpan di direktori yang benar.")
else:
    try:
        # Memuat scaler
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)

        # Membaca model
        brain_stroke_model = pickle.load(open('brain_stroke_model.sav', 'rb'))

        # Judul web
        st.title('Klasifikasi Penyakit Stroke')

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Jenis Kelamin",
                ("Perempuan", "Laki-laki"),
                index=None,
                placeholder="Pilih",
                key='gender_selectbox')

            gender_Female = 1.0 if gender == "Perempuan" else 0.0
            gender_Male = 1.0 if gender == "Laki-laki" else 0.0

        with col2:
            age = st.text_input('Usia')
            age = float(age) if age else 0.0

        with col1:
            hypertension = st.selectbox(
                "Hipertensi",
                ("Ya", "Tidak"),
                index=None,
                placeholder="Pilih",
                key='hypertension_selectbox')

            hypertension = 1 if hypertension == "Ya" else 0

        with col2:
            heart_disease = st.selectbox(
                "Penyakit Jantung",
                ("Ya", "Tidak"),
                index=None,
                placeholder="Pilih",
                key='heart_disease_selectbox')

            heart_disease = 1 if heart_disease == "Ya" else 0

        with col1:
            ever_married = st.selectbox(
                "Pernah Menikah",
                ("Ya", "Tidak"),
                index=None,
                placeholder="Pilih",
                key='ever_married_selectbox')

            ever_married = 1 if ever_married == "Ya" else 0

        with col2:
            work_type = st.selectbox(
                "Pekerjaan",
                ("Karyawan Swasta", "Wiraswasta", "Pelajar", "PNS"),
                index=None,
                placeholder="Pilih",
                key='work_type_selectbox')

            work_type_Private = 1.0 if work_type == "Karyawan Swasta" else 0.0
            work_type_Self_employed = 1.0 if work_type == "Wiraswasta" else 0.0
            work_type_children = 1.0 if work_type == "Pelajar" else 0.0
            work_type_Govt_job = 1.0 if work_type == "PNS" else 0.0

        with col1:
            Residence_type = st.selectbox(
                "Tempat Tinggal",
                ("Desa", "Kota"),
                index=None,
                placeholder="Pilih",
                key='Residence_type_selectbox')

            Residence_type = 1 if Residence_type == "Desa" else 0

        with col2:
            avg_glucose_level = st.text_input('Tingkat Glukosa')
            avg_glucose_level = float(avg_glucose_level) if avg_glucose_level else 0.0

        with col1:
            bmi = st.text_input('Indeks Massa Tubuh')
            bmi = float(bmi) if bmi else 0.0

        with col2:
            smoking_status = st.selectbox(
                "Status Merokok",
                ("Tidak pernah merokok", "Tidak diketahui", "Pernah merokok", "Merokok"),
                index=None,
                placeholder="Pilih",
                key='smoking_status_selectbox')

            smoking_status_dict = {"Tidak pernah merokok": 1, "Tidak diketahui": 0, "Pernah merokok": 2, "Merokok": 3}
            smoking_status = smoking_status_dict.get(smoking_status, 0)

        # Prediksi
        stroke_diagnosis = ''

        # Membuat tombol diagnosis
        if st.button('Test Prediksi Stroke'):
            # Membuat array input
            input_data = np.array([[age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, smoking_status, gender_Female, gender_Male, work_type_Govt_job, work_type_Private, work_type_Self_employed, work_type_children]])
            
            # Mengaplikasikan scaler
            scaled_input_data = scaler.transform(input_data)
            
            # Melakukan prediksi
            stroke_prediction = brain_stroke_model.predict(scaled_input_data)
            
            if stroke_prediction[0] == 1:
                stroke_diagnosis = 'Pasien terkena Stroke'
            else:
                stroke_diagnosis = 'Pasien tidak terkena Stroke'
            
            st.success(stroke_diagnosis)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat scaler: {e}")
