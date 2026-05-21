import streamlit as st
from dotenv import load_dotenv
import os
from main import estimate_cost


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please check the OPENAI_API_KEY.")

def scanner():
    st.title("Scanner")

    uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])
    location = st.text_input("Location")

    if st.button("Submit"):
        if uploaded_file and location:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write("Predicting food name and cost...")

            food_name, cost = estimate_cost(uploaded_file, location)
            st.subheader("Predicted Food Name")
            st.write(food_name)

            st.subheader("Estimated Home-Made Cost (per person)")
            st.write(f"{cost} USD")
        else:
            st.warning("Please upload an image and provide location.")

    
if __name__ == "__main__":
    scanner()