import streamlit as st
import pickle
import numpy as np

# ------------------------
# Load trained model
# ------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Species Prediction 🌸")
st.write("Enter the measurements of the iris flower:")

# ------------------------
# User Input
# ------------------------
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f"The predicted species is: **{prediction[0]}**")
