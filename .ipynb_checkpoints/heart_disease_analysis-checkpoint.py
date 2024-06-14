# Import all the tools we need
# Regular EDA (Exploratory Data Analysis)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay

# Load dataset
try:
    df = pd.read_csv("90 - heart-disease.csv")
except FileNotFoundError:
    print("The dataset file '90-heart-disease.csv' was not found.")
    exit()

# Top rows
print(df.head())

# Bottom rows
print(df.tail())

# Finding the number of classes
print(df['target'].value_counts())

# Plot target variable distribution
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
plt.show()

# Dataset info
print(df.info())

# Dataset description
print(df.describe())

# Compare target column with sex column
print(df.sex.value_counts())

# Crosstab between target and sex
print(pd.crosstab(df.target, df.sex))

# Plotting crosstab
pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(10, 6), color=["salmon", "lightblue"])
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.show()

# Scatter plot between age and max heart rate
plt.figure(figsize=(10, 6))
plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c="salmon")
plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c="lightblue")
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate")
plt.show()

# Histogram of age
df.age.plot.hist()
plt.show()

# Heart Disease Frequency per Chest Pain Type
pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(10, 6), color=["lightblue", "salmon"])
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["No Disease", "Disease"])
plt.show()

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
plt.show()

# Splitting the data into X and y
X = df.drop("target", axis=1)
y = df["target"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Fit and score the models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} model accuracy: {model.score(X_test, y_test):.2f}")

# Hyperparameter tuning for Logistic Regression
log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}
rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, cv=5, n_iter=20, verbose=True)
rs_log_reg.fit(X_train, y_train)
print(rs_log_reg.best_params_)
print(rs_log_reg.score(X_test, y_test))

# Random Forest hyperparameter tuning
rf_grid = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2)
}
rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True)
rs_rf.fit(X_train, y_train)
print(rs_rf.best_params_)
print(rs_rf.score(X_test, y_test))

# Grid Search for Logistic Regression
gs_log_reg = GridSearchCV(LogisticRegression(), param_grid=log_reg_grid, cv=5, verbose=True)
gs_log_reg.fit(X_train, y_train)
print(gs_log_reg.best_params_)
print(gs_log_reg.score(X_test, y_test))

# Predictions and evaluation
y_preds = gs_log_reg.predict(X_test)
print(confusion_matrix(y_test, y_preds))

# Plot confusion matrix
sns.set(font_scale=1.5)
def plot_conf_mat(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("true label")
    plt.ylabel("predicted label")

plot_conf_mat(y_test, y_preds)
plt.show()
print(classification_report(y_test, y_preds))

# Cross-validation scores
clf = LogisticRegression(C=0.23357214690901212, solver="liblinear")
cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
cv_acc = np.mean(cv_acc)
cv_precision = np.mean(cross_val_score(clf, X, y, cv=5, scoring="precision"))
cv_recall = np.mean(cross_val_score(clf, X, y, cv=5, scoring="recall"))
cv_f1 = np.mean(cross_val_score(clf, X, y, cv=5, scoring="f1"))

# Visualizing cross-validated metrics
cv_metrics = pd.DataFrame({
    "Accuracy": cv_acc,
    "Precision": cv_precision,
    "Recall": cv_recall,
    "F1": cv_f1
}, index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False)
plt.show()

# Feature importance
clf.fit(X_train, y_train)
features_dict = dict(zip(df.columns, list(clf.coef_[0])))
features_df = pd.DataFrame(features_dict, index=[0])
features_df.T.plot.bar(title="Feature Importance", legend=False)
plt.show()
