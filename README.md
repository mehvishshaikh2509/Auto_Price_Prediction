# Auto_Price_Prediction

### **1. Introduction**

The automobile industry generates a vast amount of data. Leveraging machine learning for predictive analytics helps businesses and consumers make informed decisions. In this project, we aim to build an efficient car price prediction system using various regression models, including an Artificial Neural Network (ANN).

### **2. Objective**

The primary objective of this project is to predict the price of automobiles based on several features. The project focuses on preprocessing data, handling missing values and outliers, engineering features, building various regression models, and selecting the best-performing model based on evaluation metrics.

### **3. Dataset Overview**

- Source: UCI Machine Learning Repository (Automobile dataset)

- Total Records: 200

- Features: 26 original features (numeric and categorical)

- Target Variable: Price

Key features include car specifications such as symboling, normalized-losses, wheel-base, length, width, height, curb-weight, engine-size, horsepower, peak-rpm, highway-mpg, and categorical attributes like make, fuel-type, aspiration, body-style, etc.

### **4. Methodology**

4.1 Data Preprocessing and Feature Engineering
4.1.1 Handling Missing Values

- Imputed missing numeric values using median strategies.
- Categorical missing values were filled using mode.


4.1.2 Handling Categorical Variables

- num-of-cylinders and num-of-doors were mapped to numeric values.
- One-Hot Encoding was applied to categorical columns such as make, fuel-type,                aspiration, body-style, drive-wheels, engine-location, engine-type, and fuel-system.

4.1.3 Handling Outliers

- Identified outliers in numeric features using boxplots.
- Applied the IQR (Interquartile Range) method for outlier treatment by capping extreme       values to upper and lower bounds.

4.1.4 Feature Scaling

- Used StandardScaler to scale continuous numerical features to standardize data for model    training.

4.2 Model Selection and Training

We implemented and compared the following regression models:

 1. Linear Regression

 2. K-Nearest Neighbors (KNN)

 3. Support Vector Machine (SVR)

 4. Decision Tree Regressor

 5. Random Forest Regressor

 6. Gradient Boosting Regressor

 7. XGBoost Regressor

 8. Artificial Neural Network (ANN)


### **5. Model Performance Evaluation**
**Accuracy Comparison (R2 Score & RMSE)**

| Model             | R2 Score | RMSE  |
| ----------------- | -------- | ----- |
| Linear Regression | -40.00   | 1674  |
| KNN               | 0.90     | 2634  |
| SVM               | 0.62     | 5089  |
| Decision Tree     | 0.94     | 1896  |
| Random Forest     | 0.95     | 1791  |
| Gradient Boosting | 0.96     | 1601  |
| XGBoost           | 0.95     | 1782  |
| ANN               | 0.97     | 1397  |


**Visualization**
- Bar plots were created to compare the R2 Scores and RMSE values of each model.

### **6. Best Performing Model: ANN**

The Artificial Neural Network (ANN) model emerged as the best-performing model based on both R2 Score and RMSE.

**ANN Model Architecture**

- Input Layer: 58 features

- Hidden Layers: Two layers with 64 and 32 neurons respectively, both using ReLU activation   functions

- Output Layer: One neuron with linear activation for regression output

**Hyperparameters**

- Optimizer: Adam with learning rate 0.01

- Loss Function: Mean Squared Error (MSE)

- Epochs: 100

- Batch Size: 10

**Evaluation Metrics (ANN Model)**

- Mean Absolute Error (MAE): 1148.87

- Mean Squared Error (MSE): 1986779.40

- Root Mean Squared Error (RMSE): 1409.53

- R2 Score: 0.9714

### **7. Hyperparameter Tuning & Optimization**

- For ANN: Tuned the learning rate, number of layers, and neurons.

- For Random Forest and XGBoost: Applied GridSearchCV to optimize hyperparameters such as     number of estimators, maximum depth, and learning rate.

### **8. Model Evaluation**

Train-Test Split

- Training Set: 80% (160 records)

- Testing Set: 20% (40 records)

**Cross-Validation**

- Conducted 5-Fold Cross-Validation for key models to verify consistency and minimize         overfitting risk.

**ANN Performance (Best Model)**

- R2 Score: 0.9714

- RMSE: 1409.53

### **9. Conclusion**

This project successfully developed an automobile price prediction model by applying comprehensive data preprocessing, feature engineering, and machine learning algorithms. Among all tested models, the ANN model achieved the best results with an R2 score of 0.9714 and RMSE of 1409.53.
