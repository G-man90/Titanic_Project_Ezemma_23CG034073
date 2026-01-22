import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("../data/train.csv")

# Select allowed features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'
df = df[features + [target]]

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump((model, scaler), "titanic_survival_model.pkl")

# Reload model to demonstrate persistence
loaded_model, loaded_scaler = joblib.load("titanic_survival_model.pkl")

sample = np.array([[3, 0, 25, 7.25, 0]])
sample_scaled = loaded_scaler.transform(sample)
print("Sample prediction (0=Did Not Survive, 1=Survived):",
      loaded_model.predict(sample_scaled)[0])

