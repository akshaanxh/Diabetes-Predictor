# -*- coding: utf-8 -*-


import numpy as np
import pickle 

#loading the saved model
loaded_model = pickle.load(open("C:/Users\HII\Desktop/MACHINE LEARNING/ML PROJECTS\DIABETES PREDICTION USING SVM/trained_model.sav", 'rb'))


#Making a predictive system for this model
input_data = (5,116,74,0,0,25.6,0.201,30)

#changing data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

#provide sentence outputs
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')