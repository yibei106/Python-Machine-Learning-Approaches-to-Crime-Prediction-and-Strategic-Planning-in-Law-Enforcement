# Python-Machine-Learning-Approaches-to-Crime-Prediction-and-Strategic-Planning-in-Law-Enforcement

## Project Overview

This project addresses challenges in crime prediction and resource allocation by applying machine learning techniques to the Chicago crime dataset from Kaggle. The project focuses on three key predictive tasks:
1. **Arrest Likelihood Prediction**: Predicting whether an arrest will be made for a given crime.
2. **Crime Category Prediction**: Predicting the category of crime based on basic crime features (Forbidden Practices, Theft, Assault, Public Peace Violation).
3. **Spatial Prediction**: Predicting the geographic coordinates of where a crime will occur.

By leveraging various machine learning models, data preprocessing techniques, and addressing class imbalance, the project demonstrates that meaningful predictions can be achieved using minimal information about a crime.

## Dataset

The dataset used in this project is from [Kaggle's Chicago Crime Dataset](https://www.kaggle.com/datasets/currie32/crimes-in-chicago), which includes approximately 1,000,000 reported crime incidents from 2012 to 2017. The dataset contains 23 variables that detail each crime, including the type of crime, location, date, time, and whether an arrest was made.

### Data Preprocessing

Key data preprocessing steps include:
- **Feature Engineering**: Extract day, month, time and day of the week from Date. 
- **Handling Missing Values**: Missing location values were imputed twice—first using block addresses and then using geographic coordinates. Other negligible missing values were removed.
- **Outlier Removal**: Extreme values in coordinates were identified and removed.
- **Data Transformation**: Unifying different representations of the same category.
- **Data Categorization**: Map crime types into their respective categories.
- **Feature Selection**: Remove irrelevant features.
- **Encoding**: The dataset contains both categorical and numerical variables, requiring encoding for machine learning models.
- **Train-Test Split**: The dataset was split into 70% for training and 30% for testing.

## Model Selection and Training

For each prediction task, multiple machine learning models were trained, tuned, and evaluated. Grid search and randomized search were used for hyperparameter tuning. The following models were used:

### 1. **Arrest Prediction**
   - **Random Forest**
   - **K-Nearest Neighbors (KNN)**
   - **AdaBoost** (best performer: 73.3% accuracy)
   - **Logistic Regression**

### 2. **Crime Category Prediction**
   - **Random Forest**
   - **Multilayer Perceptron (MLP)**
   - **K-Nearest Neighbors (KNN)**
   - **Naive Bayes**
   - **Gradient Boosting Models**: XGBoost, LightGBM, and CatBoost (CatBoost achieved 62% accuracy)

### 3. **Spatial Prediction (Geographic Coordinates)**
   - **Random Forest Regressor** (best performer: R-squared score of 0.997)
   - **Gradient Boosting Regressor**
   - **K-Nearest Neighbors Regressor**
   - **Polynomial Regression**

### Addressing Class Imbalance and Overfitting
- **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to balance the class distribution in tasks such as arrest prediction. (BP-SMOTE may not be as suitable for this project compared to SMOTE.)
- **Overfitting Mitigation**: Learning curves were plotted to identify and mitigate overfitting and underfitting issues.

## Deployment
The models chosen for each prediction task have been saved using **pickle** for easy loading and deployment.

The models is deployed using the **Streamlit** library, offering an interactive web interface. The deployment allows users to:
- Input crime-related data.
- Get predictions on arrest likelihood, crime category, and geographic coordinates.
- Receive recommendations based on the predictions using the **OpenAI API** for enhanced insights.

## System Interface
![image](https://github.com/user-attachments/assets/aabd4fa2-b23e-4b75-b99b-6c215eb60076)

The interface features icons that provide input descriptions when users hover over them for clarification.
![image](https://github.com/user-attachments/assets/9d7266ac-265b-4a29-88a5-3cb2b12dd6b7)

### Predictions Interface
![image](https://github.com/user-attachments/assets/d4dea46f-d006-484c-a26a-70e5cc298181)
![image](https://github.com/user-attachments/assets/26da3c06-00f3-4584-bc3b-38b3fd82da25)



