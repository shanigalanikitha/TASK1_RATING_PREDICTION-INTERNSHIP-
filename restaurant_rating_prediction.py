import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Dataset .csv")

# Display first 5 rows
print("\nFIRST 5 ROWS OF DATASET\n")
print(df.head())

# Check missing values
print("\nMISSING VALUES\n")
print(df.isnull().sum())

# Fill missing values
df.fillna("Unknown", inplace=True)

# Select Features
features = [
    'Restaurant Name',
    'City',
    'Cuisines',
    'Average Cost for two',
    'Price range',
    'Votes'
]

# Target column
target = 'Aggregate rating'

# Keep required columns
data = df[features + [target]]

# Encode categorical columns
label_encoder = LabelEncoder()

for col in ['Restaurant Name', 'City', 'Cuisines']:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Split input and output
X = data[features]
y = data[target]

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestRegressor(random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE\n")

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
accuracy = r2 * 100

print("Model Accuracy:", round(accuracy, 2), "%")

# Feature Importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE\n")
print(feature_importance)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance in Restaurant Rating Prediction")

plt.show()