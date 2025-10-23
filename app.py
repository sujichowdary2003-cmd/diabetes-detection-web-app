# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 17:33:21 2025

@author: sujic
"""

import numpy as np
import pickle
import streamlit as st

#load the saved model
loaded_model = pickle.load(open('D:\\Diabetes project\\_trained_model.sav', 'rb'))


#creating a function for Detection

def diabetes_detection(input_data):
    
    input_data = (5,166,72,19,175,8,0.587,51)
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    Detection = loaded_model.predict(input_data_reshaped)
    print(Detection)

    if (Detection[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
    
def main():
    
    #giving a title 
    st.title('Diabetes Detection ')
    
    #getting the input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose =st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    #code for Detection

    diagnosis = ''
    
    #creating a button for detection
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_detection([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__=='__main_':

    main()






