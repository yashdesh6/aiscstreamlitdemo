import streamlit as st
import pickle
import numpy as np

with open("age_height_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📏 Age vs Height Predictor")
st.write("Enter an age to predict the height.")

age = st.slider("Age (years)", 5, 18, 10)

if st.button("Predict"):
    # Perform prediction
    features = np.array([[age]])
    predicted_height = model.predict(features)[0]
    
    st.write(f"📐 **Predicted Height:** {predicted_height:.2f} cm")
