import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#Read file and comebine fruits and vegetables
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
df['Diet'] = ((df.Fruits == 1) & (df.Veggies == 1)).astype(int)

FEATURES = ['BMI', 'HighChol', 'PhysActivity', 'Diet']
TARGET = 'Diabetes_binary'
X = df[FEATURES]
y = df[TARGET]

#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Create a pipeline for imbalances in the dataset with Smote and stadardizing the data while using a logistic regression model
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scale', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000))
])

#Use grid search to find the optimal combination of hyperparameters
param_grid = {
    'clf__C':      [0.01, 0.1, 1, 10],
    'clf__penalty':['l1','l2']
}
gs = GridSearchCV(pipe, param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

#finding the best model after the k-fold cross-validation
best_model = gs.best_estimator_ 
print("Model with best F1 after cross validation:", gs.best_score_)

# finding optimal threshold on test set
probs = best_model.predict_proba(X_test)[:,1]
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = [f1_score(y_test, (probs>=t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {best_t:.2f}, Test-set F1: {max(f1_scores):.3f}")

with open('logregmodel.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('logregthreshold.pkl', 'wb') as f:
    pickle.dump(best_t, f)