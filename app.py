import streamlit as st
import pandas as pd
import os
from eda import show_basic_eda
from ml import train_and_evaluate


st.set_page_config(page_title="Open Dataset Explorer", layout="wide")

st.title("📊 Open Dataset Explorer")
st.write("Upload a CSV file to begin exploring your dataset automatically.")

# Sample dataset section
sample_path = "sample_datasets"
sample_files = [f for f in os.listdir(sample_path) if f.endswith(".csv")]

use_sample = st.checkbox("Use a sample dataset instead")
if use_sample:
    selected_sample = st.selectbox("Choose a sample dataset", sample_files)
    file_path = os.path.join(sample_path, selected_sample)
    df = pd.read_csv(file_path)
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

# Display dataset
if df is not None:
    st.success("✅ Dataset loaded successfully!")
    st.write("### Preview of Data:")
    st.dataframe(df.head(20))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    show_basic_eda(df)
    train_and_evaluate(df)
else:
    st.info("📂 Please upload a CSV file or select a sample to proceed.")
