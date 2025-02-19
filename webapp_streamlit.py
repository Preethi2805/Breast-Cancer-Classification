import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import streamlit as st

# Loading the saved model
loaded_model = load_model('E:/Personal Project/Breast Cancer/breast_cancer_model.h5') 

# Loading the scaler (replace the path with where your scaler is saved)
scaler = joblib.load('E:/Personal Project/Breast Cancer/scaler.pkl')

# Creating a function to make predictions
def breast_cancer(input_data):
    # Ensure inputs are converted to floats and handle empty inputs
    try:
        input_data = [float(i) if i else 0.0 for i in input_data]  # Convert empty inputs to 0.0
    except ValueError:
        return "Invalid input data. Please enter valid numerical values for all fields."
    
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
    if label == 0:
        return "The patient is Benign"
    else:
        return "The patient is Malignant"

# Building the webapp using Streamlit
def main():
    # Title
    st.title('Breast Cancer Classification Web App')
    
    # Getting the input data from the user
    mean_radius = st.text_input("Mean Radius of the Lobes: ")
    mean_texture = st.text_input("Mean Texture of the Lobes: ")
    mean_perimeter = st.text_input("Mean Perimeter of the Lobes: ")
    mean_area = st.text_input("Mean Area of the Lobes: ")
    mean_smoothness = st.text_input("Mean Smoothness of the Lobes: ")
    mean_compactness = st.text_input("Mean Compactness of the Lobes: ")
    mean_concavity = st.text_input("Mean Concavity of the Lobes: ")
    mean_concave_points = st.text_input("Mean Concave Points of the Lobes: ")
    mean_symmetry = st.text_input("Mean Symmetry of the Lobes: ")
    mean_fractal_dimension = st.text_input("Mean Fractal Dimension of the Lobes: ")
    
    radius_error = st.text_input("Radius Error of the Lobes: ")
    texture_error = st.text_input("Texture Error of the Lobes: ")
    perimeter_error = st.text_input("Perimeter Error of the Lobes: ")
    area_error = st.text_input("Area Error of the Lobes: ")
    smoothness_error = st.text_input("Smoothness Error of the Lobes: ")
    compactness_error = st.text_input("Compactness Error of the Lobes: ")
    concavity_error = st.text_input("Concavity Error of the Lobes: ")
    concave_points_error = st.text_input("Concave Points Error of the Lobes: ")
    symmetry_error = st.text_input("Symmetry Error of the Lobes: ")
    fractal_dimension_error = st.text_input("Fractal Dimension Error of the Lobes: ")
    
    worst_radius = st.text_input("Worst Radius of the Lobes: ")
    worst_texture = st.text_input("Worst Texture of the Lobes: ")
    worst_perimeter = st.text_input("Worst Perimeter of the Lobes: ")
    worst_area = st.text_input("Worst Area of the Lobes: ")
    worst_smoothness = st.text_input("Worst Smoothness of the Lobes: ")
    worst_compactness = st.text_input("Worst Compactness of the Lobes: ")
    worst_concavity = st.text_input("Worst Concavity of the Lobes: ")
    worst_concave_points = st.text_input("Worst Concave Points of the Lobes: ")
    worst_symmetry = st.text_input("Worst Symmetry of the Lobes: ")
    worst_fractal_dimension = st.text_input("Worst Fractal Dimension of the Lobes: ")
    
    # Prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button("Test Result"):
        diagnosis = breast_cancer([
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, 
            mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, 
            radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, 
            concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, 
            worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, 
            worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
        ])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
