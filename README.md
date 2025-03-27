# :bar_chart: **Machine Learning GUI - MKT3434 Homework-1** :computer:

Welcome to the **Machine Learning GUI** project repository! This enhanced GUI-based application allows easy access to a variety of machine learning models, data preprocessing methods, and model evaluation metrics. Developed for the MKT3434 - **Machine Learning** course, the primary goal is to simplify machine learning tasks and provide a user-friendly interface for both beginners and experienced users.

---

## :sparkles: **Features**

### 1. **Load Dataset** :file_folder:
- **Built-in Datasets**: Select from several **built-in datasets** like Iris, Breast Cancer, MNIST, and California Housing (replacing the Boston dataset).
- **Custom Dataset**: Users can **upload their own CSV datasets**. The app will prompt users to select the **target column** for supervised learning.

### 2. **Preprocess Data** :wrench:
- **Scaling**: Apply one of the following **scaling methods** to the dataset:
  - :chart_with_upwards_trend: **Standard Scaling**
  - :chart_with_downwards_trend: **Min-Max Scaling**
  - :triangular_flag_on_post: **Robust Scaling**
- **Missing Data Handling**: Choose from the following methods to handle missing data:
  - :arrows_clockwise: **Mean Imputation**: Replaces missing values with the mean of the feature.
  - :arrows_counterclockwise: **Median Imputation**: Replaces missing values with the median.
  - :arrow_forward: **Forward Fill**: Propagates the previous value forward.
  - :arrow_backward: **Backward Fill**: Propagates the next value backward.
  - :arrows_counterclockwise: **Linear Interpolation**: Fills missing values by linearly interpolating between valid values.

### 3. **Select Model and Hyperparameters** :chart_with_upwards_trend:
- **Model Selection**: Choose from regression or classification models:
  - **Regression**: Linear Regression, Support Vector Regression (SVR).
  - **Classification**: Logistic Regression, SVM, Decision Tree, Random Forest, KNN, Naive Bayes.
- **Hyperparameters**: Configure necessary hyperparameters such as `C`, `epsilon`, `kernel` for SVR, number of neighbors for KNN, etc.

### 4. **Train the Model** :muscle:
- After selecting the model and configuring the parameters, click on the **"Train"** button to begin training.
- The **training progress** is displayed through a progress bar.

### 5. **Visualize Results** :bar_chart:
- The GUI will display the performance metrics such as:
  - **Regression**: Predicted vs. Actual values in a scatter plot.
  - **Classification**: Confusion matrix and other classification metrics like accuracy, precision, recall, and F1-score.

### 6. **Evaluation Metrics** :bar_chart:
- **For Regression**: Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **For Classification**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## :warning: **Missing Data Handling Methods** :warning:

This section evaluates the impact of various missing data handling techniques on model accuracy.

### Methods Tested:
1. **Mean Imputation**: Replaces missing values with the **mean** of the available data.
2. **Median Imputation**: Replaces missing values with the **median** of the available data.
3. **Forward Fill**: Propagates the **previous valid value** forward.
4. **Backward Fill**: Propagates the **next valid value** backward.
5. **Linear Interpolation**: Estimates missing values by **linearly interpolating** between adjacent valid values.

### Results:

#### :memo: **Regression Tasks**:
- **Linear Interpolation**: Best-performing method with the **lowest MSE** and **MAE**.
- **Median Imputation**: Slightly worse, especially with **outliers**.
- **Mean Imputation**: Poor performance, especially with **outliers**.
- **Forward Fill** and **Backward Fill**: Slight improvement over Mean Imputation.

#### :memo: **Classification Tasks**:
- **Median Imputation** and **Linear Interpolation**: Achieved **high accuracy**, especially with missing feature values.
- **Mean Imputation**: **Negative impact** on classification accuracy.

### :round_pushpin: **Conclusion**:
- **For regression tasks**, **Linear Interpolation** proved to be the best, showing the lowest error metrics.
- **For classification tasks**, both **Median Imputation** and **Linear Interpolation** gave the best results, while **Mean Imputation** negatively impacted performance.

---

## :camera: **GUI Screenshots**

Here are some screenshots showing the improved GUI and how the features work:

### 1. **Data Management Section** 
![Data Management Section](screenshots/data_management.png)

### 2. **Model Training Section**
![Model Training Section](screenshots/model_training1.png)
![Model Training Section](screenshots/model_training2.png)
![Model Training Section](screenshots/model_training3.png)
![Model Training Section](screenshots/model_training4.png)

### 3. **Visualization**
![Visualization](screenshots/visualization1.png)
![Visualization](screenshots/visualization2.png)
![Visualization](screenshots/visualization3.png)
![Visualization](screenshots/visualization4.png)
![Visualization](screenshots/visualization5.png)
![Visualization](screenshots/visualization6.png)

---

## :rocket: **Requirements**

To run the application, make sure you have the following dependencies installed:

- Python 3.x
- PyQt6
- Matplotlib
- Scikit-learn
- TensorFlow

You can install all dependencies by running:

```bash
pip install -r requirements.txt
