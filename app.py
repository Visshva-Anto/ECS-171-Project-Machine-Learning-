from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained Neural Network model and scaler
with open('nnmodel.pkl', 'rb') as f:
    nn_model = pickle.load(f)
with open('nnscaler.pkl', 'rb') as f:
    nn_scaler = pickle.load(f)

# Load logistic regression model and threshold
with open('logregmodel.pkl', 'rb') as f:
    logreg_model = pickle.load(f)
with open('logregthreshold.pkl', 'rb') as f:
    logreg_threshold = pickle.load(f)

# Load decision tree model and threshold
with open('dtreemodel.pkl', 'rb') as f:
    dtree_model = pickle.load(f)
with open('dtreethreshold.pkl', 'rb') as f:
    dtree_threshold = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    bmi = float(request.form['bmi'])               
    cholesterol = int(request.form['cholesterol'])  # 0 = normal, 1 = high
    activity = int(request.form['activity'])        # 0 = no, 1 = yes
    diet = int(request.form['diet'])                # 0 = poor, 1 = healthy

    # Store predictions for all models
    predictions = {}

    # Neural Network prediction
    nn_input = np.array([[bmi, cholesterol, activity, diet]])
    nn_scaled = nn_scaler.transform(nn_input)
    nn_pred = nn_model.predict(nn_scaled)[0]
    predictions['Neural Network'] = "Yes" if nn_pred == 1 else "No"

    # Logistic Regression prediction
    logreg_input = np.array([[bmi, cholesterol, activity, diet]])
    logreg_scaled = logreg_model.named_steps['scale'].transform(logreg_input)
    logreg_prob = logreg_model.named_steps['clf'].predict_proba(logreg_scaled)[0][1]
    logreg_pred = int(logreg_prob >= logreg_threshold)
    predictions['(Best Model) Logistic Regression'] = f"{'Yes' if logreg_pred else 'No'} ({logreg_prob:.2f})"

    # Decision Tree prediction
    tree_input = np.array([[bmi, cholesterol, activity, diet]])
    tree_prob = dtree_model.predict_proba(tree_input)[0][1]
    tree_pred = int(tree_prob >= dtree_threshold)
    predictions['Decision Tree'] = f"{'Yes' if tree_pred else 'No'} ({tree_prob:.2f})"

    return render_template('form.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
