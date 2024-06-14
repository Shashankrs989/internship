import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('90 - heart-disease.csv')

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Logistic Regression
logreg = LogisticRegression(solver='liblinear', max_iter=10000)
logreg_grid = {'C': np.logspace(-4, 4, 20), 'solver': ['liblinear']}
logreg_cv = GridSearchCV(logreg, logreg_grid, cv=5)
logreg_cv.fit(X_train_scaled, y_train)
joblib.dump(logreg_cv, 'logreg_model.pkl')

# KNN
knn = KNeighborsClassifier()
knn_grid = {'n_neighbors': np.arange(1, 21)}
knn_cv = GridSearchCV(knn, knn_grid, cv=5)
knn_cv.fit(X_train_scaled, y_train)
joblib.dump(knn_cv, 'knn_model.pkl')

# Random Forest
rf = RandomForestClassifier()
rf_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_cv = GridSearchCV(rf, rf_grid, cv=5)
rf_cv.fit(X_train_scaled, y_train)
joblib.dump(rf_cv, 'rf_model.pkl')

print("Models and scaler have been trained and saved as 'logreg_model.pkl', 'knn_model.pkl', 'rf_model.pkl', and 'scaler.pkl'.")
