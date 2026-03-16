import pandas as pd                     
import matplotlib.pyplot as plt         
from sklearn.model_selection import train_test_split   
from sklearn.linear_model import LinearRegression      
from sklearn.metrics import mean_squared_error, r2_score  

# Task1: Regression Analysis
# Load the dataset
df = pd.read_csv("HousePrediction.csv", sep='\s+')
print(df.columns)
# Display the first 5 rows of the dataset
print("First 5 rows of dataset:")
print(df.head())

# Check dataset information
# This helps verify data types and missing values
print("\nDataset Information:")
print(df.info())

# Select features independent variable and target dependent variable
# We will use 'open' price to predict 'close' price
X = df.iloc[:, [0]]  
y = df.iloc[:, 13]       


# Split the dataset into training and testing sets
# 80% for training the model and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", len(X_train))
print("Testing data size:", len(X_test))

# Create the Linear Regression model
model = LinearRegression()

# Train fit the model using the training data
model.fit(X_train, y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
# Mean Squared Error 
mse = mean_squared_error(y_test, y_pred)

# R-squared score (R²)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2 Score):", r2)

# Display the regression coefficient and intercept
print("\nModel Parameters:")
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# Visualize the regression results
# Scatter plot of actual data
plt.scatter(X_test, y_test)

# Plot the regression line
plt.plot(X_test, y_pred)

# Add labels and title
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.title("Linear Regression: Predicting Close Price from Open Price")

# Show the plot
plt.show()
