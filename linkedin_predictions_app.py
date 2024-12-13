import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Function to clean binary features
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Load and preprocess the dataset
data = pd.read_csv("social_media_usage.csv")
data['sm_li'] = clean_sm(data['web1h'])
data['income'] = np.where(data['income'] <= 9, data['income'], np.nan)
data['educ2'] = np.where(data['educ2'] <= 8, data['educ2'], np.nan)
data['par'] = clean_sm(data['par'])
data['marital'] = clean_sm(data['marital'])
data['female'] = clean_sm(data['gender'] == 2)
data['age'] = np.where(data['age'] <= 98, data['age'], np.nan)

df = data[['sm_li', 'income', 'educ2', 'par', 'marital', 'female', 'age']].dropna()
X = df[['income', 'educ2', 'par', 'marital', 'female', 'age']]
y = df['sm_li']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("LinkedIn Usage Predictor")
st.header("Predict if a person uses LinkedIn and how likely they use it")

# User input
st.markdown("####1 being the lowest Income/Education and 9 being the highest Income/Education")
income = st.slider("Income (1-9):", 1, 9, 5)
educ2 = st.slider("Education (1-8)", 1, 8, 4)
par = st.selectbox("Are you a parent?", ["No", "Yes"])
marital = st.selectbox("Married?", ["No", "Yes"])
female = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age (18-97)", 18, 97, 30)

# Convert input
par = 1 if par == "Yes" else 0
marital = 1 if marital == "Yes" else 0
female = 1 if female == "Female" else 0

# Prediction
input_features = [[income, educ2, par, marital, female, age]]
probability = model.predict_proba(input_features)[0][1]
prediction = model.predict(input_features)[0]

st.write("Prediction: ", "LinkedIn User" if prediction == 1 else "Non-User")
st.write(f"Probability of being a LinkedIn user: {probability:.2%}")
