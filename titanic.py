import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# LOAD DATA

df=pd.read_excel("C:\\Users\\Janiffer\\OneDrive\\문서\\INTERNSHIP\\streamlit_app\\titanic.xlsx")

st.title("Titanic Survival Prediction App")
st.write("Predict whether a passenger survived the Titanic disaster")


# DATA CLEANING

# Drop useless columns
df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

# Fill missing Age and Fare
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Fill missing Embarked
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode Sex
df['Sex'] = df['Sex'].map({'male':0,'female':1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# MODEL TRAINING

X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X[['Age','Fare']] = scaler.fit_transform(X[['Age','Fare']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "titanic_model.joblib")

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader(f"Model Accuracy: {round(acc*100,2)}%")


# USER INPUT SECTION

st.header("Passenger Details")

pclass = st.selectbox("Ticket Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
sex = 0 if sex=="male" else 1

age = st.number_input("Age", min_value=0.0, max_value=90.0, step=1.0)
sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children", min_value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, step=1.0)
embarked = st.selectbox("Port of Embarkation", ["S","C","Q"])

# Create Embarked dummy variables
embarked_S = 1 if embarked=="S" else 0
embarked_Q = 1 if embarked=="Q" else 0
# (C becomes baseline automatically)

# Scale age and fare
age = (age - df['Age'].mean()) / df['Age'].std()
fare = (fare - df['Fare'].mean()) / df['Fare'].std()


# PREDICTION

if st.button("Predict Survival"):
    model = joblib.load("titanic_model.joblib")

    input_data = np.array([[pclass, sex, age, sibsp, parch, fare,
                            embarked_Q, embarked_S]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Passenger SURVIVED")
    else:
        st.error("Passenger DID NOT SURVIVE")
st.write("App started successfully")