# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:43:04 2025

@author: sibas
"""

import numpy as np
import pickle
import streamlit as st

# Use a raw string (r"...") to avoid issues with backslashes
model_path = r"D:\5. Course\FREE YT courses\Siddardhan_ML\Diabetes deployment\trained_model.sav"

# Load the model
loaded_model = pickle.load(open(model_path, 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    
    
    # converting input data to numpay array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #giving title for our webpage
    st.title('Diabetes prediction web page')
    
    #getting the input from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Thickness of skin')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function')
    Age = st.text_input('Age of the person')
    
    
    # codeing part for prediction
    diagnosis = ''
    
    #creating a button for prediction 
    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    