import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\chandan\Downloads\patients_data.xlsx"   
df = pd.read_excel(file_path)

# Display first few rows
print(df.head())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Histograms for numerical variables
numerical_cols = ["age", "height", "weight", "bmi"]
df[numerical_cols].hist(figsize=(12, 6), bins=20, edgecolor="black")
plt.suptitle("Histograms of Numerical Variables")
plt.show()

# Count plots for categorical variables
categorical_cols = ["assigned_sex", "state", "country"]
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Frequency of {col}')
    plt.show()

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Boxplot of BMI by Assigned Sex
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["assigned_sex"], y=df["bmi"])
plt.title("BMI Distribution by Assigned Sex")
plt.show()

# Scatter plot of Height vs. Weight
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["height"], y=df["weight"], hue=df["assigned_sex"])
plt.title("Height vs Weight")
plt.show()
