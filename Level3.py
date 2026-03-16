import pandas as pg
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Task 1: Predictive Modeling(Classification)
# Read the CSV file "churn.csv" into a pandas DataFrame called 'dfg'
dfg = pg.read_csv("churn.csv")

# Print the entire DataFrame to see all the data
print(dfg)

# Print the first 5 rows of the DataFrame
# Useful for inspecting the data structure and values
print(dfg.head())

# Display dataset information
print("\nDataset information:")
print(dfg.info())

# Print the DataFrame to check for missing values
print("\nMissing values in dataset:")
print(dfg.isnull().sum())

# Encoder categorical variables
# Machine leaning models require numeric values
# Convert text columns into numeric form
le = LabelEncoder()

for column in dfg.select_dtypes(include=['object']).columns:
    dfg[column] = le.fit_transform(dfg[column])
    
# Separate features and target variable
# Target column = churn and Feature = all other columns   
x = dfg.drop("Churn",axis=1)
y = dfg["Churn"]  

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Feature scalling
# Standardize features to improve model performance  
scaler = StandardScaler()
 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Logistic Regression model
log_model = LogisticRegression()

log_model.fit(x_train, y_train)
log_pred = log_model.predict(x_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test,log_pred))

print("\nLogistic Regression Report:")
print(classification_report(y_test,log_pred))

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test,dt_pred))

print("\nDecision Tree Report:")
print(classification_report(y_test,dt_pred))

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(x_train, y_train)
rf_model = rf_model.predict(x_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test,dt_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test,dt_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix-Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_estimators':[50, 100, 200],
              'max_depth':[5, 10, 20]
              }

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(x_train, y_train)

print("\nBest Parameters Found:")
print(grid.best_params_)

# Evaluate tuned Found
best_model = grid.best_estimator
best_pred = best_model.predict(x_test)

print("\nTuned Random Forest Accuracy:")
print(accuracy_score(y_test,best_pred))
