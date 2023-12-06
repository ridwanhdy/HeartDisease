# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:53:45 2022

@author: Benk
"""
import streamlit as st
# import polars as pl
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pickle
from sklearn.linear_model import LogisticRegression



# pip list --format=freeze
# streamlit run "C:\Users\Benk\Desktop\PowerBi\Personal Key Indicators of Heart Disease\main.py"

st.image("banner.png", use_column_width=True)

data = pd.read_csv("heart_2020_cleaned.csv")
st.subheader('Visualisasi Data Utama :')
st.dataframe(data)


st.write("""
         ### Untuk memprediksi status penyakit jantung Anda:
         ###### 1- Masukkan parameter yang paling menggambarkan Anda.
         ###### 2- Tekan tombol "Prediksi" dan tunggu hasilnya.
         """)

# Load the data
# st.write(BMIdata)
# -------------------------------------------------------------------------
st.sidebar.title('Silakan isi informasi Anda untuk memprediksi kondisi jantung Anda')



BMI=st.sidebar.selectbox("Pilih BMI Kamu", ("Normal weight BMI  (18.5-25)", 
                             "Underweight BMI (< 18.5)" ,
                             "Overweight BMI (25-30)",
                             "Obese BMI (> 30)"))
Age=st.sidebar.selectbox("Pilih Umur Kamu", 
                            ("18-24", 
                             "25-29" ,
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "55-59",
                             "80 or older"))

Race=st.sidebar.selectbox("Pilih Ras Kamu", ("Asian", 
                             "Black" ,
                             "Hispanic",
                             "American Indian/Alaskan Native",
                             "White",
                             "Other"
                             ))

Gender=st.sidebar.selectbox("Pilih Jenis Kelamin", ("Female", 
                             "Male" ))
Smoking = st.sidebar.selectbox("Pernahkah Kamu Merokok Lebih Dari 100 Batang Rokok"
                          " Sepanjang Hidupmu ?)",
                          options=("No", "Yes"))
alcoholDink = st.sidebar.selectbox("Apakah kamu sering Minuman Beralkohol dalam Seminggu?", options=("No", "Yes"))
stroke = st.sidebar.selectbox("Apakah Kamu Punya Stroke?", options=("No", "Yes"))

sleepTime = st.sidebar.number_input("Jumlah Tidur Dalam 24h", 0, 24, 7) 

genHealth = st.sidebar.selectbox("General health",
                             options=("Good","Excellent", "Fair", "Very good", "Poor"))

physHealth = st.sidebar.number_input("Kesehatan jasmani dalam sebulan terakhir (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
mentHealth = st.sidebar.number_input("Kesehatan mental dalam sebulan terakhir (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
physAct = st.sidebar.selectbox("Aktivitas fisik dalam sebulan terakhir"
                           , options=("No", "Yes"))



diffWalk = st.sidebar.selectbox("Apakah Anda mengalami kesulitan serius dalam berjalan"
                            " Atau menaiki tangga?", options=("No", "Yes"))
diabetic = st.sidebar.selectbox("Apakah Anda pernah menderita diabetes?",
                           options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
asthma = st.sidebar.selectbox("Apakah Anda menderita asma?", options=("No", "Yes"))
kidneyDisease= st.sidebar.selectbox("Apakah Anda menderita penyakit ginjal?", options=("No", "Yes"))
skinCancer = st.sidebar.selectbox("Apakah Anda menderita kanker kulit?", options=("No", "Yes"))

dataToPredic = pd.DataFrame({
   "BMI": [BMI],
   "Smoking": [Smoking],
   "AlcoholDrinking": [alcoholDink],
   "Stroke": [stroke],
   "PhysicalHealth": [physHealth],
   "MentalHealth": [mentHealth],
   "DiffWalking": [diffWalk],
   "Gender": [Gender],
   "AgeCategory": [Age],
   "Race": [Race],
   "Diabetic": [diabetic],
   "PhysicalActivity": [physAct],
   "GenHealth": [genHealth],
   "SleepTime": [sleepTime],
   "Asthma": [asthma],
   "KidneyDisease": [kidneyDisease],
   "SkinCancer": [skinCancer]
 })

#-------------------------Mapping-------------------------------
dataToPredic.replace("Underweight BMI (< 18.5)",0,inplace=True)
dataToPredic.replace("Normal weight BMI  (18.5-25)",1,inplace=True)
dataToPredic.replace("Overweight BMI (25-30)",2,inplace=True)
dataToPredic.replace("Obese BMI (> 30)",3,inplace=True)

dataToPredic.replace("Yes",1,inplace=True)
dataToPredic.replace("No",0,inplace=True)
dataToPredic.replace("18-24",0,inplace=True)
dataToPredic.replace("25-29",1,inplace=True)
dataToPredic.replace("30-34",2,inplace=True)
dataToPredic.replace("35-39",3,inplace=True)
dataToPredic.replace("40-44",4,inplace=True)
dataToPredic.replace("45-49",5,inplace=True)
dataToPredic.replace("50-54",6,inplace=True)
dataToPredic.replace("55-59",7,inplace=True)
dataToPredic.replace("60-64",8,inplace=True)
dataToPredic.replace("65-69",9,inplace=True)
dataToPredic.replace("70-74",10,inplace=True)
dataToPredic.replace("75-79",11,inplace=True)
dataToPredic.replace("80 or older",13,inplace=True)


dataToPredic.replace("No, borderline diabetes",2,inplace=True)
dataToPredic.replace("Yes (during pregnancy)",3,inplace=True)


dataToPredic.replace("Excellent",0,inplace=True)
dataToPredic.replace("Good",1,inplace=True)
dataToPredic.replace("Fair",2,inplace=True)
dataToPredic.replace("Very good",3,inplace=True)
dataToPredic.replace("Poor",4,inplace=True)


dataToPredic.replace("White",0,inplace=True)
dataToPredic.replace("Other",1,inplace=True)
dataToPredic.replace("Black",2,inplace=True)
dataToPredic.replace("Hispanic",3,inplace=True)
dataToPredic.replace("Asian",4,inplace=True)
dataToPredic.replace("American Indian/Alaskan Native",4,inplace=True)


dataToPredic.replace("Female",0,inplace=True)
dataToPredic.replace("Male",1,inplace=True)


filename='finalized_model.sav'
loaded_model= pickle.load(open(filename, 'rb'))
Result=loaded_model.predict(dataToPredic)
ResultProb= loaded_model.predict_proba(dataToPredic)
ResultProb1=round(ResultProb[0][1] * 100, 2)

if st.button('PREDICT'):
 # st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
 if (ResultProb1>30):
  st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
 else:
  st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
  
  
  
  

