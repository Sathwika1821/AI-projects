import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load dataset
housing = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["HousePrice"] = housing.target

# Features (X) and Target (y)
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check error
mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", mae)

# Compare actual and predicted prices
results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": predictions[:10]
})

print("\nFirst 10 Predictions:")
print(results)
# Scatter plot
plt.scatter(y_test, predictions)

# Labels
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

# Show graph
plt.show()
import joblib

joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")