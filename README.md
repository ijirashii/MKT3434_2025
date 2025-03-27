# Machine Learning GUI - MKT3434 Homework-1

This repository contains an enhanced version of a GUI-based application for machine learning, developed for MKT3434 - Machine Learning course. The primary goal of this project is to enable easy access to a variety of machine learning models, data preprocessing methods, and model evaluation metrics.

## Features

### 1. **Load Dataset**
- **Built-in Datasets**: Select from several built-in datasets such as Iris, Breast Cancer, MNIST, and California Housing (replacing the Boston dataset).
- **Custom Dataset**: Users can upload their own CSV datasets. The app will prompt users to select the target column for supervised learning.

### 2. **Preprocess Data**
- **Scaling**: Apply scaling methods to the dataset using the following options:
  - Standard Scaling
  - Min-Max Scaling
  - Robust Scaling
- **Missing Data Handling**: Select and apply one of the following methods to handle missing data:
  - **Mean Imputation**: Replaces missing values with the mean of the non-missing values of the feature.
  - **Median Imputation**: Replaces missing values with the median of the non-missing values of the feature.
  - **Forward Fill**: Fills missing values by propagating the previous value forward.
  - **Backward Fill**: Fills missing values by propagating the next value backward.
  - **Linear Interpolation**: Fills missing values by linearly interpolating between adjacent values.

### 3. **Select Model and Hyperparameters**
- **Model Selection**: Choose a regression or classification model from the respective tabs:
  - **Regression**: Linear Regression, Support Vector Regression (SVR).
  - **Classification**: Logistic Regression, Support Vector Machine (SVM), Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Naive Bayes.
- **Hyperparameters**: Configure necessary hyperparameters for each model such as `C`, `epsilon`, `kernel` for SVR, number of neighbors for KNN, etc.

### 4. **Train the Model**
- After selecting the model and configuring the parameters, click on the **"Train"** button to start the training process.
- The progress bar will show the current training status.

### 5. **Visualize Results**
- The GUI will display the following based on the task type:
  - **For Regression**: Predicted vs. Actual values in a scatter plot.
  - **For Classification**: Confusion matrix, along with performance metrics like accuracy, precision, recall, and F1-score.

### 6. **Evaluation Metrics**
- **For Regression**: Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **For Classification**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## Missing Data Handling Methods

This section evaluates the impact of different missing data handling techniques on model accuracy.

### Methods Tested:
1. **Mean Imputation**: Replaces missing values with the mean of the available data in that feature.
2. **Median Imputation**: Replaces missing values with the median of the available data.
3. **Forward Fill**: Propagates the previous valid value forward to fill the missing value.
4. **Backward Fill**: Propagates the next valid value backward to fill the missing value.
5. **Linear Interpolation**: Estimates missing values by linearly interpolating between adjacent valid values.

### Results:

#### Regression Tasks:
- **Linear Interpolation**: Best-performing method with the lowest MSE and MAE.
- **Median Imputation**: Performed slightly worse, especially with outliers.
- **Mean Imputation**: Showed poor performance, especially in datasets with outliers.
- **Forward Fill** and **Backward Fill**: Performed similarly, showing slight improvement over Mean Imputation.

#### Classification Tasks:
- **Median Imputation** and **Linear Interpolation**: Both achieved high accuracy, especially in datasets with missing feature values.
- **Mean Imputation**: Had a noticeable negative impact on classification accuracy, especially with skewed datasets.

### Conclusion:
- **For regression tasks**, Linear Interpolation is the most effective method, yielding the lowest error metrics.
- **For classification tasks**, Median Imputation and Linear Interpolation provide the best results, while Mean Imputation negatively impacts performance.

---

## GUI Screenshots

Here are some screenshots showing the improved GUI and how the features work:

### 1. Data Management Section
![Data Management Section](screenshots/data_management.png)

### 2. Model Training Section
![Model Training Section](screenshots/model_training1.png)
![Model Training Section](screenshots/model_training2.png)
![Model Training Section](screenshots/model_training3.png)
![Model Training Section](screenshots/model_training4.png)

### 3. Visualization
![Visualization](screenshots/visualization1.png)
![Visualization](screenshots/visualization2.png)
![Visualization](screenshots/visualization3.png)
![Visualization](screenshots/visualization4.png)
![Visualization](screenshots/visualization5.png)

---

## Requirements

To run the application, make sure you have the following dependencies installed:

- Python 3.x
- PyQt6
- Matplotlib
- Scikit-learn
- TensorFlow

You can install all dependencies by running:

```bash
pip install -r requirements.txt