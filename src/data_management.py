
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data
def load_heart_failure_data():
    """
    Load the Heart Failure dataset.
    """

    data_path = (
        PROJECT_ROOT
        / "outputs"
        / "datasets"
        / "collection"
        / "HeartFailureData.csv"
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    return df


@st.cache_resource
def load_pkl_file(file_path):
    """
    Load a trained machine learning model or pipeline.
    """

    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {file_path}"
        )

    return joblib.load(file_path)
