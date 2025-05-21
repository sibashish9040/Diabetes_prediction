# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
# Use a raw string (r"...") to avoid issues with backslashes
model_path = r"D:\5. Course\FREE YT courses\Siddardhan_ML\Diabetes deployment\trained_model.sav"

# Load the model
loaded_model = pickle.load(open(model_path, 'rb'))

input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
# converting input data to numpay array
input_data_as_numpy_array = np.asarray(input_data)
# reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')