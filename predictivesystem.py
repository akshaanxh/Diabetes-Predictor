# -*- coding: utf-8 -*-
import numpy as np
import joblib
from sklearn.svm import SVC  # or the actual model you trained
from sklearn.datasets import load_iris  # Replace with your diabetes dataset if needed

<<<<<<< HEAD
#loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))
=======
# 🚨 Step 1: Train a simple SVM model (skip if already trained)
# This is only for demonstration; replace it with your real trained model.
# X, y = your diabetes data and labels
# model = SVC()
# model.fit(X, y)
>>>>>>> beb986f (Fix: update app.py with relative path and formatting)

# joblib.dump(model, 'trained_model.sav')  # Save only after training once

# ✅ Step 2: Load the pre-trained model
loaded_model = joblib.load('trained_model.sav')

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
<<<<<<< HEAD
  print('The person is diabetic')
=======
    print("The person is diabetic")
>>>>>>> beb986f (Fix: update app.py with relative path and formatting)
