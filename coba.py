import pickle
import streamlit as st

# Membaca model
brain_stroke_model = pickle.load(open('brain_stroke_model.sav', 'rb'))

# Judul web
st.title('Klasifikasi Penyakit Stroke')

# Membagi layout menjadi dua kolom
col1, col2 = st.columns(2)

# Input untuk Jenis Kelamin (Gender)
with col1:
    gender = st.selectbox(
        "Jenis Kelamin",
        ("Perempuan", "Laki-laki"),
        index=None,
        placeholder="Pilih",
        key='gender_selectbox'
    )
    if gender is not None:
        gender_Female = 1 if gender.lower() == "perempuan" else 0
        gender_Male = 1 if gender.lower() == "laki-laki" else 0

# Input untuk Usia (Age)
with col2:
    age = st.text_input('Usia')

# Input untuk Hipertensi (Hypertension)
with col1:
    hypertension = st.selectbox(
        "Hipertensi",
        ("Ya", "Tidak"),
        index=None,
        placeholder="Pilih",
        key='hypertension_selectbox'
    )
    hypertension = 1 if hypertension.lower() == "ya" else 0 if hypertension.lower() == "tidak" else None

# Input untuk Penyakit Jantung (Heart Disease)
with col2:
    heart_disease = st.selectbox(
        "Penyakit Jantung",
        ("Ya", "Tidak"),
        index=None,
        placeholder="Pilih",
        key='heart_disease_selectbox'
    )
    heart_disease = 1 if heart_disease.lower() == "ya" else 0 if heart_disease.lower() == "tidak" else None

# Input untuk Pernah Menikah (Ever Married)
with col1:
    ever_married = st.selectbox(
        "Pernah Menikah",
        ("Ya", "Tidak"),
        index=None,
        placeholder="Pilih",
        key='ever_married_selectbox'
    )
    ever_married = 1 if ever_married.lower() == "ya" else 0 if ever_married.lower() == "tidak" else None

# Input untuk Pekerjaan (Work Type)
with col2:
    work_type = st.selectbox(
        "Pekerjaan",
        ("Karyawan Swasta", "Wiraswasta", "Pelajar", "PNS"),
        index=None,
        placeholder="Pilih",
        key='work_type_selectbox'
    )
    if work_type is not None:
        work_type_Private = 1 if work_type.lower() == "karyawan swasta" else 0
        work_type_Self_employed = 1 if work_type.lower() == "wiraswasta" else 0
        work_type_children = 1 if work_type.lower() == "pelajar" else 0
        work_type_Govt_job = 1 if work_type.lower() == "pns" else 0

# Input untuk Tempat Tinggal (Residence Type)
with col1:
    residence_type = st.selectbox(
        "Tempat Tinggal",
        ("Desa", "Kota"),
        index=None,
        placeholder="Pilih",
        key='residence_type_selectbox'
    )
    Residence_type = 0 if residence_type.lower() == "kota" else 1 if residence_type.lower() == "desa" else None

# Input untuk Tingkat Glukosa (Average Glucose Level)
with col2:
    avg_glucose_level = st.text_input('Tingkat Glukosa')

# Input untuk Indeks Massa Tubuh (BMI)
with col1:
    bmi = st.text_input('Indeks Massa Tubuh')

# Input untuk Status Merokok (Smoking Status)
with col2:
    smoking_status = st.selectbox(
        "Status Merokok",
        ("Tidak pernah merokok", "Tidak diketahui", "Pernah merokok", "Merokok"),
        index=None,
        placeholder="Pilih",
        key='smoking_status_selectbox'
    )
    if smoking_status is not None:
        smoking_status = 1 if smoking_status.lower() == "tidak pernah merokok" else 0 if smoking_status.lower() == "tidak diketahui" else 2 if smoking_status.lower() == "pernah merokok" else 3

# Tombol untuk melakukan prediksi
if st.button('Test Prediksi Stroke'):
    # Lakukan prediksi
    stroke_prediction = brain_stroke_model.predict([[age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, smoking_status, gender_Female, gender_Male, work_type_Govt_job, work_type_Private, work_type_Self_employed, work_type_children]])
    
    # Tampilkan hasil prediksi
    if stroke_prediction[0] == 1:
        stroke_diagnosis = 'Pasien terkena Stroke'
    else:
        stroke_diagnosis = 'Pasien tidak terkena Stroke'
    st.success(stroke_diagnosis)
