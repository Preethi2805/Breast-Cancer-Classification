import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # To load the scaler

# Load the saved model
loaded_model = load_model('E:/Personal Project/Breast Cancer/breast_cancer_model.h5')

# Load the saved scaler (Ensure you have saved it as a .pkl file)
scaler = joblib.load('E:/Personal Project/Breast Cancer/scaler.pkl')

# Input data for a single patient
input_data = (19.81, 22.15, 130, 1260, 0.09831, 0.1027, 0.1479, 0.09498, 0.1582, 0.05395, 
              0.7582, 1.017, 5.865, 112.4, 0.006494, 0.01893, 0.03391, 0.01521, 0.01356, 0.001997, 
              27.32, 30.88, 186.8, 2398, 0.1512, 0.315, 0.5372, 0.2388, 0.2768, 0.07615)

# Convert input data to numpy array & reshape
input_array = np.asarray(input_data).reshape(1, -1)

# Standardize the input data
input_array_standard = scaler.transform(input_array)

# Convert input to TensorFlow tensor
input_tensor = tf.convert_to_tensor(input_array_standard, dtype=tf.float32)

# Predict
prediction = loaded_model.predict(input_tensor)

# Extract class label (0 = Benign, 1 = Malignant)
label = np.argmax(prediction, axis=1)[0]

# Output result
print("Prediction:", prediction)
print("Label:", label)
if label == 0:
    print("The patient is Benign")
else:
    print("The patient is Malignant")
