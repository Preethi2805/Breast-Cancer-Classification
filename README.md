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

---

## 🚀 Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/breast-cancer-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd breast-cancer-classification
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Run Logistic Regression Classification**:
   ```bash
   python logistic_regression_model.py
   ```
2. **Run Neural Network Classification**:
   ```bash
   python neural_network_model.py
   ```
3. **Make Predictions**:
   Modify the `input_data` in the script to provide your test case. Example:
   ```python
   input_data = (19.81, 22.15, 130, 1260, 0.09831, 0.1027, 0.1479, ...)
   ```

---

## 📊 Project Workflow

1. **Data Loading**:
   - Data collected from the Scikit-learn Breast Cancer dataset.
   - Includes 30 features describing tumor characteristics.
2. **Exploratory Data Analysis**:
   - Summary statistics, feature distributions, and correlation analysis.
3. **Preprocessing**:
   - Standardization for neural network training.
   - Outlier detection and handling.
4. **Model Training**:
   - Logistic Regression for baseline performance.
   - Neural Networks for advanced performance.
5. **Prediction System**:
   - Predict whether a patient’s condition is benign or malignant based on input features.

---

## 📂 Project Structure

```
breast-cancer-classification/
├── logistic_regression_model.py   # Logistic Regression implementation
├── neural_network_model.py        # Neural Network implementation
├── data_visualization.py          # Data visualization scripts
├── breast_cancer.csv              # Dataset
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
```

---

## 🤝 Contributions

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
