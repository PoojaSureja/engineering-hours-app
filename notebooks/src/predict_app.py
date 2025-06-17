
import streamlit as st
import requests
import numpy as np


databricks_token = "dapi6b51821ac8550f016bcc7d609b143f22"

# ✅ Correct way to use the secret value
DATABRICKS_TOKEN = st.secrets["databricks_token"]

st.title("Engineering Hours Predictor")
st.write("Enter the values below to predict engineering hours")

def inverse_boxcox(y, lmbda):
    import numpy as np
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.power(y * lmbda + 1, 1 / lmbda)


# ✅ Replace with actual lambda used in training
BOXCOX_LAMBDA = 0.10350704191564682

# Input fields
stock_count = st.number_input("Stock Count", value=5)
unique_model_count = st.number_input("Unique Model Count", value=2)
repeted_customer_count = st.number_input("Repeated Customer Count", value=1)
total_price = st.number_input("Total Price (per stock)", value=10000.0)
ind_2020 = st.selectbox("Project began after January 2020?", options=[0, 1])
customer_no = st.text_input("Customer Number", value="C1234")

# Predict button
if st.button("Predict"):
    url = "https://adb-3010693782896111.11.azuredatabricks.net/serving-endpoints/yyy/invocations"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",  # ✅ Use Bearer and actual token value
        "Content-Type": "application/json"
    }

    payload = {
        "dataframe_split": {
            "columns": ["unique_model_count", "repeted_customer_count", "ind_2020", "total_price", "stock_count", "customer_no"],
            "data": [[unique_model_count, repeted_customer_count, ind_2020, total_price, stock_count, customer_no]]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        # ✅ Extract Box-Cox transformed prediction
        transformed_pred = result.get("predictions", [None])[0]

        if transformed_pred is not None:
            # 🔁 Inverse Box-Cox transform
            original_pred = inverse_boxcox(transformed_pred, BOXCOX_LAMBDA)  # ✅ CORRECT
            st.success(f"Predicted Engineering Hours: {original_pred:.2f}")
        else:
            st.error("Prediction could not be retrieved from response.")
    except Exception as e:
            st.error(f"Error occurred: {str(e)}")
