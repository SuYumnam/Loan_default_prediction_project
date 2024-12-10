# **Loan default Prediction using this machine learning model**

This project use various machine learning techniques to predict the risk of loan default using a financial loan services dataset from kaggle. By identifying individuals at a higher risk of default, the main objective of our project is to assist loan service companies in making informed decisions and reducing financial risks.

---
## Table of Contents
 1.  [Introduction](#Introduction).
 2.  [Project Objectives](#Project_Objectives)
 3.  [Methodologies](#Methodologies)
 4.  [Project Structruce](#Project_Structure)
 5.  [Set up](#Set_up)
 6.  [Technologies Used](#Technologies_Used)
 7.  [Dataset](#Dataset)
 8.  [How to used](#How_to_used)  
 9.  [Implementation](#Implementation)
 10. [Model Evaluation and validation](#Model_Evaluation_and_validation)
 11. [Model Deployment](#Model_Deployment)

---
## **Introduction**

The main goal of this project is to develop the best machine learning model to predict loan default risk. This analysis is important for loan service companies or institution to mitigate risks and improve efficiency in loan management

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
 ## **Project Structrure**
 
- **data/**: Contains the dataset.
  - `Loan_default.csv`: The CSV file with loan data.

- **notebooks/**: Contains Jupyter notebooks for data preparation, model selection, and API deployment.
  - `data_preparation_and_model_selection.ipynb`: Notebook for data preparation, dataset exploration, model training, and selection.
  - `api_deployment.ipynb`: Notebook for configuring and testing the Flask API using ngrok.

- **models/**: Contains the trained model.
  - `best_knn_model.pkl`: The trained model file saved with `joblib`.

- **app/**: Contains the Flask API code.
  - `app.py`: The Python file with the Flask API code.

- **requirements.txt**: Lists all dependencies and libraries required for the project.

- **README.md**: Project documentation.
  
  ---
  ## **Set up**
  1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/loan_default_prediction.git
   cd loan_default_prediction

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

3. Start the Flask API:
   ```bash
   python app/app.py

4. Access the API using the provided ngrok URL

 
 ---
## **Technologies Used**
- Python: pandas, numpy, matplotlib, seaborn, scikit-learn. 
- Machine Learning Algorithms: Logistic Regression, Random Forest, Gradient Boosting, KNN, Stacking, and Voting Classifiers
- Visualization: Correlation matrix and histograms
- Environment: Anaconda, Jupyter Notebook, Python
- Deployment: Flask API, FlaskNgrok, Joblib
---
## **Dataset**
Source of the dataset [Loan Default dataset from kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default?select=Loan_default.csv)
Attributes: 
- Numeric: Age, Income, Loan Amount, etc.
- Categorical: Employment Type, Marital Status, Loan Purpose, etc.
---
## **How to used**

1. Open the Jupyter notebooks in the notebooks/ directory to explore the data, train models, and see the model selection process.
2.  Use the app.py file in the app/ directory to start the Flask API and make predictions.

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
## **Model Evaluation and Validation**

A detailed comparison of performance of all the models are provided in the code. Metrics include accuracy, precision, recall, and F1 score and the final model was selected based on its overall performance.
### Best Models by Metric

- **Accuracy**: Best Stacking Classifier con 0.887547
- **Precision**: Best Voting Classifier con 0.756098
- **Recall**: Stacking Classifier con 0.096441
- **F1 Score**: Stacking Classifier con 0.164641

### Results of model comparison

| Model                      | Accuracy | Precision | Recall  | F1 Score |
|----------------------------|----------|-----------|---------|----------|
| Logistic Regression        | 0.885882 | 0.621622  | 0.031186| 0.059393 |
| Best Logistic Regression   | 0.885549 | 0.611336  | 0.025593| 0.049130 |
| Random Forest              | 0.886802 | 0.630769  | 0.048644| 0.090323 |
| Best Random Forest         | 0.886078 | 0.713542  | 0.023220| 0.044977 |
| Gradient Boosting          | 0.887312 | 0.656587  | 0.051525| 0.095552 |
| Best Gradient Boosting     | 0.887468 | 0.603239  | 0.075763| 0.134618 |
| K-Nearest Neighbors        | 0.874858 | 0.324266  | 0.076780| 0.124161 |
| Best K-Nearest Neighbors   | 0.883415 | 0.444898  | 0.036949| 0.068232 |
| Voting Classifier          | 0.886000 | 0.629139  | 0.032203| 0.061271 |
| Best Voting Classifier     | 0.884883 | 0.756098  | 0.005254| 0.010436 |
| Stacking Classifier        | 0.886939 | 0.562253  | 0.096441| 0.164641 |
| Best Stacking Classifier   | 0.887547 | 0.604527  | 0.076949| 0.136521 |

---

## **Model Deployment**
- The Flask API is implemented in the api_deployment.ipynb notebook and the app.py file.

- The API receives input data through a web form and returns predictions on whether a loan will default or not.


