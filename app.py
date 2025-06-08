from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Neural Network model
with open('nnmodel.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler from the Neural Network model
with open('nnscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    bmi = float(request.form['bmi'])
    cholesterol = int(request.form['cholesterol'])  # 0 = normal, 1 = high
    activity = int(request.form['activity'])        # 0 = no, 1 = yes
    diet = int(request.form['diet'])                # 0 = poor, 1 = healthy

    features = np.array([[bmi, cholesterol, activity, diet]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    result = 'Yes' if prediction[0] == 1 else 'No'

    # Debugging thingy (can remove later)
    print("Input:", features)
    print("Scaled:", features_scaled)
    print("Prediction:", result)

    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
