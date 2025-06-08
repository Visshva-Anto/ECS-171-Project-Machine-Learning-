import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
df['Diet'] = ((df['Fruits'] == 1) & (df['Veggies'] == 1)).astype(int)

FEATURES = ['BMI', 'HighChol', 'PhysActivity', 'Diet']
TARGET = 'Diabetes_binary'
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='sgd',
    alpha=1e-3,
    learning_rate_init=1e-3,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-5,
    random_state=42
)
mlp.fit(X_resampled, y_resampled)

X_test_scaled = scaler.transform(X_test)
y_pred = mlp.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open('nnmodel.pkl', 'wb') as f:
    pickle.dump(mlp, f)

with open('nnscaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)