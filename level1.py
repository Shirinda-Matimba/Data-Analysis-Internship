import pandas as pg
import matplotlib.pyplot as plt

# Task 1: Data Cleaning and Prepocessing
# Description is to work with raw dataset that contains missing values, duplicates and incosistent data formats

# Read the CSV file "iris.csv" into a pandas DataFrame called 'dfg'
dfg = pg.read_csv("iris.csv")

# Print the entire DataFrame to see all the data
print(dfg)

# Print the first 5 rows of the DataFrame
# Useful for quickly inspecting the data structure and values
print(dfg.head())

# Check for missing values in each column
# isnull().sum() returns the number of missing values per column
print(dfg.isnull().sum())

# Check for duplicate rows in the DataFrame
# duplicated().sum() returns the total number of duplicate rows
print(dfg.duplicated().sum())

# Display the actual duplicate rows in the DataFrame
# Filters the DataFrame to show only rows that are duplicates
print(dfg[dfg.duplicated()])

# Attempt to remove duplicate rows and print the result
# Using inplace=True modifies 'dfg' directly and returns None
# So this will print 'None'
print(dfg.drop_duplicates(inplace=True))

# Properly remove duplicate rows from the DataFrame (in-place)
dfg.drop_duplicates(inplace=True)

# Print the DataFrame after removing duplicates to confirm duplicates are gone
print(dfg)


# Select the first 5 rows for plotting
df_first5 = dfg.head(5)
print("First 5 rows:\n", df_first5)

# Task 3: Basic Data Visualization
# Description is to create basic plots and charts to visualize the distribution and relashinships within dataset

# Line Plot: Sepal Length vs Petal Length (first 5 rows)
plt.figure(figsize=(6,4))
plt.plot(df_first5['sepal_length'], df_first5['petal_length'], color='green')
plt.title('Line Plot: Sepal Length vs Petal Length (First 5 Rows)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.tight_layout()
plt.savefig('line_first5.png') 
plt.show()
