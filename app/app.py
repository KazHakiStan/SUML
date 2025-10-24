import streamlit as st
from predict import predict

st.title("Iris Species Prediction")

st.write("Enter the following features of the Iris flower:")

sepal_length = st.number_input('Sepal Length (cm)', min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.1, max_value=2.5, step=0.1)

if st.button('Predict'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    predicted_species = predict(features)

    st.write(f"Predicted Iris Species: **{predicted_species}**")
