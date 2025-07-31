# Employee Attrition Analysis and Prediction

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/HR-Employee-Attrition.csv"
df = pd.read_csv(url)

# Basic info
print("Dataset shape:", df.shape)
print(df.info())
print(df['Attrition'].value_counts())

# Encode target variable (Attrition: Yes=1, No=0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Check missing values
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)

# Distribution of target variable
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Distribution')
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Analyze numeric features by Attrition
numeric_features = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole']

for feature in numeric_features:
    plt.figure(figsize=(8,4))
    sns.kdeplot(df.loc[df['Attrition'] == 0, feature], label='No Attrition')
    sns.kdeplot(df.loc[df['Attrition'] == 1, feature], label='Attrition')
    plt.title(f'Distribution of {feature} by Attrition')
    plt.legend()
    plt.show()

# Encode categorical variables
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('Attrition') if 'Attrition' in categorical_features else None

le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# Prepare data for modeling
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefficients['AbsCoefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=coefficients.head(10), x='Coefficient', y='Feature', palette='viridis')
plt.title('Top 10 Features Influencing Attrition')
plt.show()
