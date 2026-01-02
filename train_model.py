import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load dataset
df = pd.read_csv("data/car_data.csv")

# 2. Select only required columns (freeze feature scope)
df = df[
    ["year", "km_driven", "fuel", "transmission", "owner", "selling_price"]
]

# 3. Clean categorical columns
df["fuel"] = df["fuel"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["transmission"] = df["transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

df["owner"] = df["owner"].map({
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4
})

# 4. Drop rows with missing or unmapped values
df.dropna(inplace=True)

# 5. Split features & target
X = df[["year", "km_driven", "fuel", "transmission", "owner"]]
y = df["selling_price"]

# 6. Train-test split (for sanity, even if not exposed yet)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Save trained model
with open("model/car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
