import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
df['Diet'] = ((df.Fruits == 1) & (df.Veggies == 1)).astype(int)

FEATURES = ['BMI', 'HighChol', 'PhysActivity', 'Diet']
TARGET = 'Diabetes_binary'
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
])

param_grid = {
    'clf__max_depth':         [3, 5, 7, 10, None],
    'clf__min_samples_split': [2, 10, 20, 50],
    'clf__min_samples_leaf':  [1, 5, 10, 20],
    'clf__ccp_alpha':         [0.0, 0.001, 0.005, 0.01]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_model  = grid.best_estimator_
best_params = grid.best_params_

probs      = best_model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 101)
f1_scores  = [f1_score(y_test, probs >= t) for t in thresholds]
best_idx   = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

# Save the trained Decision Tree pipeline
with open('dtreemodel.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the best threshold
with open('dtreethreshold.pkl', 'wb') as f:
    pickle.dump(best_thresh, f)
