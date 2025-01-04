import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load datasets
categories = pd.read_csv("C:\Project_Git\Prediction Analysis\categories.csv")
items = pd.read_csv("C:\Project_Git\Prediction Analysis\items.csv")
shops = pd.read_csv("C:\Project_Git\Prediction Analysis\shops.csv")

# Display a few rows
print("Categories:\n", categories.head())
print("Items:\n", items.head())
print("Shops:\n", shops.head())

# Merge items with categories
items_with_categories = pd.merge(items, categories, on="category_id", how="left")

# Add synthetic shop data (if shop-specific details are available)
shops['avg_sales'] = np.random.randint(50, 500, size=len(shops))  # Example shop-level feature

# Display merged data
print("Items with Categories:\n", items_with_categories.head())
print("Shops with Features:\n", shops.head())

# Add synthetic sales data for demonstration purposes
np.random.seed(42)
items_with_categories['sales'] = np.random.randint(1, 100, size=len(items_with_categories))

# Add features
items_with_categories['item_length'] = items_with_categories['item_name'].str.len()  # Length of item name
items_with_categories['is_popular_category'] = items_with_categories['category_name'].apply(
    lambda x: 1 if x in ["Electronics", "Clothing"] else 0)  # Popular category
print("Feature Engineered Data:\n", items_with_categories.head())

# Features and target
X = items_with_categories[['item_length', 'is_popular_category']]  # Features
y = items_with_categories['sales']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Features:\n", X_train.head())
print("Training Target:\n", y_train.head())

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.grid()
plt.show()

items_with_categories.to_csv(r"C:/Project_Git/Prediction Analysis/final_data.csv", index=False)
print("Final data saved to data/final_data.csv")

