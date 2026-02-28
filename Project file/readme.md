The project is in the form of file.
📁 Project Structure

Create folder like this:

FraudDetectionApp/
│
├── app.py
├── model.pkl   (your trained model file)
│
├── templates/
│     ├── index.html
│     └── result.html
│
└── static/
      └── style.css

✅ STEP 1: Install Required Libraries

Open terminal and run:
pip install flask numpy pandas scikit-learn

✅ STEP 2: Create train_model.py

Create new file: train_model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Generate dummy training data
data = {
    'step': np.random.randint(1, 100, 200),
    'type': np.random.randint(0, 2, 200),
    'amount': np.random.randint(100, 10000, 200),
    'oldbalanceOrg': np.random.randint(0, 50000, 200),
    'newbalanceOrig': np.random.randint(0, 50000, 200),
    'oldbalanceDest': np.random.randint(0, 50000, 200),
    'newbalanceDest': np.random.randint(0, 50000, 200),
    'isFraud': np.random.randint(0, 2, 200)
}

df = pd.DataFrame(data)

X = df.drop('isFraud', axis=1)
y = df['isFraud']

model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully!")

▶ STEP 3: Run Training File (VERY IMPORTANT)

In terminal:
python train_model.py

You should see:
Model trained and saved successfully!
Now model.pkl will appear in your folder ✅

✅ STEP 4: app.py (FINAL WORKING VERSION)

Replace your app.py with this:

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        step = float(request.form['step'])
        type_val = float(request.form['type'])
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Arrange features
        features = np.array([[step, type_val, amount,
                              oldbalanceOrg, newbalanceOrig,
                              oldbalanceDest, newbalanceDest]])

        # Make prediction
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Fraud"
        else:
            result = "Not Fraud"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

✅ STEP 5: templates/index.html

Create folder templates
Inside create index.html
<!DOCTYPE html>
<html>
<head>
    <title>Online Payments Fraud Detection</title>
    <style>
        body {
            font-family: Arial;
            background-color: #cfe8f9;
            text-align: center;
        }

        .container {
            width: 50%;
            margin: auto;
            margin-top: 50px;
            padding: 30px;
            background-color: #b3d9f2;
            border-radius: 10px;
        }

        h2 {
            margin-bottom: 25px;
        }

        .form-group {
            margin: 15px 0;
        }

        input {
            width: 60%;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }

        button {
            margin-top: 20px;
            padding: 10px 25px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
        }

        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>

<body>

<div class="container">
    <h2>Online Payments Fraud Detection</h2>

    <form action="/predict" method="post">

        <div class="form-group">
            <label>Step</label><br>
            <input type="text" name="step" required>
        </div>

        <div class="form-group">
            <label>Type (0 or 1)</label><br>
            <input type="text" name="type" required>
        </div>

        <div class="form-group">
            <label>Amount</label><br>
            <input type="text" name="amount" required>
        </div>

        <div class="form-group">
            <label>OldbalanceOrg</label><br>
            <input type="text" name="oldbalanceOrg" required>
        </div>

        <div class="form-group">
            <label>NewbalanceOrig</label><br>
            <input type="text" name="newbalanceOrig" required>
        </div>

        <div class="form-group">
            <label>OldbalanceDest</label><br>
            <input type="text" name="oldbalanceDest" required>
        </div>

        <div class="form-group">
            <label>NewbalanceDest</label><br>
            <input type="text" name="newbalanceDest" required>
        </div>

        <button type="submit">Predict</button>

    </form>
</div>

</body>
</html>

✅ STEP 6: templates/result.html
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Result</title>
    <style>
        body {
            font-family: Arial;
            background-color: #cfe8f9;
            text-align: center;
        }

        .container {
            width: 50%;
            margin: auto;
            margin-top: 100px;
            padding: 40px;
            background-color: #b3d9f2;
            border-radius: 10px;
        }

        .result {
            font-size: 22px;
            color: red;
            font-weight: bold;
        }

        a {
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: blue;
        }
    </style>
</head>

<body>

<div class="container">
    <h2>Online Payments Fraud Detection</h2>

    <p>The predicted fraud for the online payment is:</p>

    <div class="result">
        {{ prediction }}
    </div>

    <a href="/">Go Back</a>

</div>

</body>
</html>

✅ STEP 7: static/style.css

Create folder static
Inside create style.css
body {
    font-family: Arial;
    background-color: #cde6f5;
    text-align: center;
}

.container {
    width: 40%;
    margin: auto;
    margin-top: 60px;
    padding: 30px;
    background: #b3d7ee;
    border-radius: 10px;
}

input {
    width: 80%;
    padding: 8px;
    margin: 8px;
    border-radius: 5px;
    border: none;
}

button {
    padding: 10px 20px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

button:hover {
    background-color: #0056b3;
}

.result {
    color: red;
    font-weight: bold;
}

🚀 FINAL RUN

After training:
python app.py

Open browser:
http://127.0.0.1:5000/

Flask app is running successfully ✅

Now you must do these steps 👇
✅ STEP 1: Open the Website

Open your browser (Chrome / Edge).

Type this in address bar:
http://127.0.0.1:5000

Press Enter.

👉 Your Fraud Detection webpage should open.

✅ STEP 2: Test the Prediction

Enter some sample values like:

Field	Example Value
Step	10
Type	1
Amount	5000
OldbalanceOrg	10000
NewbalanceOrig	5000
OldbalanceDest	20000
NewbalanceDest	25000

Then click Predict.

👉 It will show:
Fraud
  or
Not Fraud

✅ STEP 3: Stop the Server (When Finished)

In VS Code terminal press:
CTRL + C
That will stop Flask.

📖 Project Description

The Online Payments Fraud Detection System is a Machine Learning-based web application developed to detect fraudulent online transactions.

In today’s digital world, online payments are increasing rapidly, and fraud activities are also growing. This project helps in identifying whether a transaction is fraudulent or legitimate based on transaction details.

The system uses a Machine Learning algorithm (Logistic Regression) to analyze transaction features such as:

Step

Transaction Type

Amount

Old Balance of Sender

New Balance of Sender

Old Balance of Receiver

New Balance of Receiver

The model is trained using sample transaction data and saved using Pickle.
A Flask web application is used to create a user-friendly interface where users can enter transaction details and get real-time fraud prediction.

🎯 Objective

The main objective of this project is:

To detect fraudulent online transactions.

To reduce financial risks.

To provide a simple web interface for prediction.

🛠 Technologies Used

Python

Flask

NumPy

Pandas

Scikit-learn

HTML

CSS

⚙️ How It Works

The user enters transaction details in the web form.

The input data is sent to the Flask backend.

The trained Machine Learning model predicts whether the transaction is:

Fraud

Not Fraud

The result is displayed on the webpage.

🚀 Key Features

✔ Real-time fraud prediction
✔ User-friendly web interface
✔ Machine Learning-based decision system
✔ Simple and lightweight design

Team ID : LTVIP2025TMID47042

Team Size : 4

Team Leader : Kalavala Pranavi

Team member : G Supraja

Team member : G Sowmys

Team member : Haritha Paaladhi.
