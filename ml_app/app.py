import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load model
model = load_model("model")

# Load dataset for feature reference
iris = load_iris(as_frame=True)
feature_names = iris.feature_names
target_names = iris.target_names

# Load comparison (optional)
@st.cache_data
def load_comparison():
    try:
        return pd.read_csv("model_comparison.csv")
    except:
        return None

comparison_df = load_comparison()

# Streamlit layout
st.sidebar.title("🧪 Iris Classifier")
page = st.sidebar.radio("Choose page", ["🧬 Predict", "📊 Compare Models"])

# Page: Predict
if page == "🧬 Predict":
    st.title("🌸 Iris Species Prediction")

    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                               columns=feature_names)

    if st.button("Predict"):
        prediction = predict_model(model, data=input_data)
        pred_class = int(prediction["prediction_label"][0])
        st.success(f"Predicted species: **{target_names[pred_class]}**")

# Page: Model Comparison
elif page == "📊 Compare Models":
    st.title("📊 Model Comparison (Accuracy)")

    if comparison_df is not None:
        st.dataframe(comparison_df)
        st.bar_chart(comparison_df.set_index("Model")["Accuracy"])
    else:
        st.warning("No comparison data available.")
