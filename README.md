# **Loan default Prediction using this machine learning model**

This project use various machine learning techniques to predict the risk of loan default using a financial loan services dataset from kaggle. By identifying individuals at a higher risk of default, the main objective of our project is to assist loan service companies in making informed decisions and reducing financial risks.

---
## Table of Contents
 1.  [Introduction](#Introduction).
 2.  [Project Objectives](#Project_Objectives)
 3.  [Methodologies](#Methodologies)
 4.  [Technologies Used](#Technologies_Used)
 5.  [Dataset](#Dataset)
 6.  [Implementation](#Implementation)
 7.  [Model Comparison](#Model_Comparison)
 8.  [Expected Outcome](#Expected_Outcome)
 9.  [Model Deployment](#Model_Deployment)

---
## **Introduction**

The main goal of this project is to develop the best machine learning model to predict loan default risk. This analysis is important for loan service companies or institution to mitigate risks and improve efficiency in loan management.

---
## **Project Objectives**
1. Dataset Analysis: Explore and preprocess the data.
2. Model Comparison: - Train and evaluate multiple machine learning models:
                     - Logistic Regression
                     - Random Forest
                     - Gradient Boosting
                     - K-Nearest Neighbors
                     - Voting Classifier
3. Hyperparameter Tuning: Optimize model performance using techniques like GridSearchCV and RandomizedSearchCV.
4. Functional Model Implementation: Develop a real-world applicable REST API predictive model using flask.
---
## **Methodology**
1. Data Preprocessing:

     - Cleaning and encoding categorical variables.
     - Standardizing numerical features.
     
2. Model Training and Evaluation:

     - Use cross-validation for performance evaluation.
       
3. Metrics:
   
     - Accuracy, Precision, Recall, F1 Score.
       
4. Ensembling Techniques:
   
     - Use methods like StackingClassifier to combine models for improved accuracy.
       
5. Hyperparameter Tuning:
   
    - Optimize parameters using GridSearchCV and RandomizedSearchCV.
 ---
## **Technologies Used**
- Python: pandas, numpy, matplotlib, seaborn, scikit-learn. 
- Machine Learning Algorithms: Logistic Regression, Random Forest, Gradient Boosting, KNN, Stacking, and Voting Classifiers
- Visualization: Correlation matrix and histograms
- Environment: Anaconda, Jupyter Notebook, Python
- Deployment: Flask API
---
## **Dataset**
Source of the dataset [Loan Default dataset from kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default?select=Loan_default.csv)
Attributes: 
- Numeric: Age, Income, Loan Amount, etc.
- Categorical: Employment Type, Marital Status, Loan Purpose, etc.
---
## **Implementation**
1. Data Preparation:

- Imported required libraries.
- Preprocessed and scaled numerical features.
- Encoded categorical data.
- Checked for null values and duplicates

2. Model Training:
- Applied various machine learning models.
- Conducted hyperparameter tuning.
  
3. Evaluation:
- Validated models using test data.
- Compared metrics for model selection.

4. Models Implemented:

- Logistic Regression (Baseline and Optimized)
- Random Forest (Baseline and Optimized)
- Gradient Boosting (Baseline and Optimized)
- K-Nearest Neighbors (Baseline and Optimized)
- Voting Classifier
- Stacking Classifier
---
## **Model Comparison**

A detailed comparison of performance of all the models are provided in the code. Metrics include accuracy, precision, recall, and F1 score and the final model was selected based on its overall performance.

---
## **Expected Outcomes**
- The expectations by the end of this project are as follows:
  
   - A highly accurate and reliable machine learning model for predicting loan defaults.
   - Insights into the features contributing to loan default.
   - Practical implications for improving loan risk managemet.

