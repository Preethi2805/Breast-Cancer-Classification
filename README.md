# 🩺 Breast Cancer Classification

This project leverages **Logistic Regression** and **Neural Networks** to classify breast cancer cases as either malignant or benign using the **Breast Cancer Dataset** from the Scikit-learn library. It demonstrates the power of machine learning in healthcare to assist in early diagnosis and decision-making.

---

## 📋 Features

- **Dataset**: Breast Cancer Dataset from Scikit-learn with 569 samples and 30 features.
- **Techniques Used**:
  - Logistic Regression for quick and interpretable classification.
  - Neural Networks for improved accuracy and performance.
- **Data Preprocessing**:
  - Standardization of features.
  - Removal of unnecessary columns.
  - Handling class imbalance.
- **Visualizations**:
  - Feature distributions.
  - Correlation matrix to identify relationships between features.
  - Outlier detection using box plots.
- **Performance**:
  - Logistic Regression:
    - Training Accuracy: `94.7%`
    - Test Accuracy: `95.6%`
  - Neural Network:
    - Training Accuracy: `96.6%`
    - Test Accuracy: `95.6%`
  - Neural network performance improves with each epoch, reducing loss and increasing accuracy.

**Web Application (Streamlit):**  
This project also features a Streamlit-based web app, where users can input values to predict whether a tumor is benign or malignant. The web app provides an easy interface to interact with the model.

**Files Added:**
- **breast_cancer_model.h5**  
- **breast_cancer_predictive.py**  
- **scaler.pkl**  
- **webapp_streamlit.py**

### Streamlit Web App

Here is a screenshot of the Streamlit web app where users can interact with the models:

![Streamlit App]([images/streamlit_app_screenshot.png)](https://github.com/Preethi2805/Breast-Cancer-Classification/blob/main/Steamlit_interface)

📊 **Project Workflow**  
- **Data Loading:** Data collected from the Scikit-learn Breast Cancer dataset.  
  - Includes 30 features describing tumor characteristics.  
- **Exploratory Data Analysis:** Summary statistics, feature distributions, and correlation analysis.  
- **Preprocessing:**  
  - Standardization for neural network training.  
  - Outlier detection and handling.  
- **Model Training:**  
  - Logistic Regression for baseline performance.  
  - Neural Networks for advanced performance.  
- **Prediction System:**  
  - Predict whether a patient’s condition is benign or malignant based on input features.  
  - Streamlit app allows users to easily interact with the model and get predictions.

---

## 🤝 Contributions

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
