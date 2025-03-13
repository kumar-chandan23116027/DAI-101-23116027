import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Cleaning
# Load dataset
df = pd.read_excel("patient_data.xlsx")

# Inspect dataset
print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Handle missing values
df.fillna(df.mean(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Detect & Handle Outliers (IQR Method)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
df = df.mask(outlier_mask, np.nan)
df.fillna(df.mean(), inplace=True)

# Standardize categorical values
df = df.apply(lambda x: x.str.lower().str.strip() if x.dtype == "object" else x)

# Step 2: Exploratory Data Analysis (EDA)
# Univariate Analysis
numerical_cols = ["age", "height", "weight", "bmi"]
df[numerical_cols].hist(figsize=(12, 6), bins=20, edgecolor="black")
plt.suptitle("Histograms of Numerical Variables")
plt.show()

for col in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col])
    plt.title(f'Frequency of {col}')
    plt.show()

# Bivariate Analysis
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["assigned_sex"], y=df[col])
    plt.title(f'{col} Distribution by Assigned Sex')
    plt.show()

sns.scatterplot(x=df["height"], y=df["weight"], hue=df["assigned_sex"])
plt.title("Height vs Weight")
plt.show()

# Multivariate Analysis
sns.pairplot(df[numerical_cols])
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Multivariate Correlation Heatmap")
plt.show()
   


