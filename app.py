#!/usr/bin/env python
# coding: utf-8

# In[8]:


from flask import Flask, request, jsonify, render_template_string
import joblib

# Load the trained model
model = joblib.load('../models/stacking_classifier_random_search.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Loan Default Prediction</title>
    </head>
    <body>
        <h1>Loan Default Prediction</h1>
        <form action="/predict" method="post">
            <label for="Age">Age:</label>
            <input type="number" id="Age" name="Age" required><br><br>

            <label for="Income">Income:</label>
            <input type="number" id="Income" name="Income" required><br><br>

            <label for="LoanAmount">Loan Amount:</label>
            <input type="number" id="LoanAmount" name="LoanAmount" required><br><br>

            <label for="CreditScore">Credit Score:</label>
            <input type="number" id="CreditScore" name="CreditScore" required><br><br>

            <label for="MonthsEmployed">Months Employed:</label>
            <input type="number" id="MonthsEmployed" name="MonthsEmployed" required><br><br>

            <label for="NumCreditLines">Number of Credit Lines:</label>
            <input type="number" id="NumCreditLines" name="NumCreditLines" required><br><br>

            <label for="InterestRate">Interest Rate:</label>
            <input type="number" step="0.01" id="InterestRate" name="InterestRate" required><br><br>

            <label for="LoanTerm">Loan Term (months):</label>
            <input type="number" id="LoanTerm" name="LoanTerm" required><br><br>

            <label for="DTIRatio">DTI Ratio:</label>
            <input type="number" step="0.01" id="DTIRatio" name="DTIRatio" required><br><br>

            <label for="Education">Education Level:</label>
            <select id="Education" name="Education" required>
                <option value="0">High School</option>
                <option value="1">Bachelor's</option>
                <option value="2">Master's</option>
                <option value="3">PhD</option>
            </select><br><br>

            <label for="EmploymentType">Employment Type:</label>
            <select id="EmploymentType" name="EmploymentType" required>
                <option value="0">Full-time</option>
                <option value="1">Part-time</option>
                <option value="2">Self-employed</option>
                <option value="3">Unemployed</option>
            </select><br><br>

            <label for="MaritalStatus">Marital Status:</label>
            <select id="MaritalStatus" name="MaritalStatus" required>
                <option value="0">Single</option>
                <option value="1">Married</option>
                <option value="2">Divorced</option>
                <option value="3">Widowed</option>
            </select><br><br>

            <label for="HasMortgage">Has Mortgage:</label>
            <select id="HasMortgage" name="HasMortgage" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br><br>

            <label for="HasDependents">Has Dependents:</label>
            <select id="HasDependents" name="HasDependents" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br><br>

            <label for="LoanPurpose">Loan Purpose:</label>
            <select id="LoanPurpose" name="LoanPurpose" required>
                <option value="0">Home</option>
                <option value="1">Car</option>
                <option value="2">Education</option>
                <option value="3">Business</option>
                <option value="4">Other</option>
            </select><br><br>

            <label for="HasCoSigner">Has Co-Signer:</label>
            <select id="HasCoSigner" name="HasCoSigner" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select><br><br>

            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [
        int(data['Age']),
        float(data['Income']),
        float(data['LoanAmount']),
        int(data['CreditScore']),
        int(data['MonthsEmployed']),
        int(data['NumCreditLines']),
        float(data['InterestRate']),
        int(data['LoanTerm']),
        float(data['DTIRatio']),
        int(data['Education']),
        int(data['EmploymentType']),
        int(data['MaritalStatus']),
        int(data['HasMortgage']),
        int(data['HasDependents']),
        int(data['LoanPurpose']),
        int(data['HasCoSigner'])
    ]
    prediction = model.predict([features])
    if prediction[0] == 1:
        result = "The applicant is likely to default on the loan."
    else:
        result = "The applicant is likely to repay the loan."
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Loan Default Prediction Result</title>
    </head>
    <body>
        <h1>Prediction Result</h1>
        <p>{{ prediction }}</p>
        <button onclick="window.location.href='/'">Go Back</button>
    </body>
    </html>
    ''', prediction=result)

if __name__ == '__main__':
    app.run()


# In[ ]:




