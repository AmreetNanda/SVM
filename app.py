import streamlit as st
from Support_Vector_Machine.inference import predict

st.title("Total Bill Prediction using SVR (Tips Dataset)")
st.write("Enter the details to predict the total bill:")

tip = st.number_input("Tip Amount", min_value=0.0, step=0.1)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.number_input("Group Size", min_value=1, max_value=10)

if st.button("Predict Total Bill"):
    result = predict(tip, sex, smoker, day, time, size)
    st.success(f"Predicted Total Bill: ${result:.2f}")
