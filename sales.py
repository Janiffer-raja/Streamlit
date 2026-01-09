import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("C:\Users\Janiffer\OneDrive\문서\INTERNSHIP\streamlit_app\clean_cafe_sales.csv")

st.title("Sales Prediction App")
st.write("Predict whether spending is HIGH or LOW")

# -----------------------------
# Data preprocessing
# -----------------------------

# Create target variable
df["High_Spend"] = (df["Total Spent"] > df["Total Spent"].median()).astype(int)

# Drop ID column
df = df.drop("Transaction ID", axis=1)

# Encode categorical columns
cat_cols = ["Item", "Payment Method", "Location"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Convert date to useful feature
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
df["Month"] = df["Transaction Date"].dt.month
df = df.drop("Transaction Date", axis=1)

# Split data
X = df.drop("High_Spend", axis=1)
y = df["High_Spend"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "sales_model.joblib")

# -----------------------------
# Streamlit inputs
# -----------------------------
st.subheader("Enter Transaction Details")

item = st.selectbox("Item", sorted(df["Item"].unique()))
payment = st.selectbox("Payment Method", sorted(df["Payment Method"].unique()))
location = st.selectbox("Location", sorted(df["Location"].unique()))
quantity = st.number_input("Quantity", min_value=1, step=1)
price = st.number_input("Price Per Unit", min_value=0.0, step=0.5)
month = st.slider("Month", 1, 12)

total_spent = quantity * price

st.write(f"Calculated Total Spent: {total_spent}")

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Spending Level"):
    model = joblib.load("sales_model.joblib")
    input_data = np.array([[item, quantity, price, total_spent, payment, location, month]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("High Spending Customer")
    else:
        st.info("Low Spending Customer")
