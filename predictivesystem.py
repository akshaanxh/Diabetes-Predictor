# -*- coding: utf-8 -*-
import numpy as np
import pickle
from sklearn.svm import SVC  # or the actual model you trained
from sklearn.datasets import load_iris  # Replace with your diabetes dataset if needed


#loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

# joblib.dump(model, 'trained_model.sav')  # Save only after training once

# ✅ Step 3: Make predictions
input_data = (5, 116, 74, 0, 0, 25.6, 0.201, 30)

# Convert to NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape for single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Predict using the loaded model
prediction = loaded_model.predict(input_data_reshaped)

# Output result
if prediction[0] == 0:
    print("The person is not diabetic")
else:

    print("The person is diabetic")
