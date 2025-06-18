
import streamlit as st
import requests
import numpy as np
import pandas as pd


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


# Load the lookup table once
def load_repeat_counts():
    return pd.read_csv("repeted_customer_counts.csv")

import os
@st.cache_data
def load_repeat_counts():
    file_path = os.path.join(os.path.dirname(__file__), "repeted_customer_counts.csv")
    return pd.read_csv(file_path)


# Input fields
stock_count = st.number_input("Stock Count", value=5)
Unique_Model_Count = st.number_input("Unique Model Count", value=2)
total_price_total = st.number_input("Total Price", value=50000.0)
total_price = total_price_total / stock_count if stock_count != 0 else 0
ind_2020 = st.selectbox("Project began after January 2020?", options=[0, 1])
CustomerNo = st.text_input("Customer Number", value="C1234")
# Automatically look up repeated count
# Automatically look up repeated count
if CustomerNo in repeat_dict:
    repeted_customer_count = repeat_dict[CustomerNo]
    st.metric(label="Repeated Customer Count (auto-calculated)", value=repeted_customer_count)
else:
    repeted_customer_count = 0
    st.text_input("Repeated Customer Count (autoCalculate)", value="0", disabled=True)
    st.warning("Customer number not found in historical data. Assuming repeated count = 0.")


# Predict button
if st.button("Predict"):
    url = "https://adb-3010693782896111.11.azuredatabricks.net/serving-endpoints/yyy/invocations"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",  # ✅ Use Bearer and actual token value
        "Content-Type": "application/json"
    }

    payload = {
        "dataframe_split": {
            "columns": ["Unique_Model_Count", "repeted_customer_count", "ind_2020", "total_price", "stock_count", "CustomerNo"],
            "data": [[Unique_Model_Count, repeted_customer_count, ind_2020, total_price, stock_count, CustomerNo]]
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
