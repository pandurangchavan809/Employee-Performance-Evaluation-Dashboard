import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
import matplotlib as plt

# Ensure model folder exists
if not os.path.exists('model'):
    os.makedirs('model')

# Load sample dataset
data = pd.read_csv('employee_data.csv')

# Features & target
X = data[['attendance', 'task_efficiency', 'teamwork', 'initiative', 'project_quality']]
y = data['performance_score']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/performance_model.pkl')
print("✅ Model trained and saved as performance_model.pkl")
