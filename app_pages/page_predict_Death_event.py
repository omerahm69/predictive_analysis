
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def page_predict_Death_event_body():

    version = "v1"

    model_dir = (
        PROJECT_ROOT
        / "outputs"
        / "ml_pipeline"
        / "predict_Death_event"
        / version
    )

    st.title("ML Pipeline: Predict Death Event")

    # -----------------------------------
    # Load fitted pipelines
    # -----------------------------------

    death_event_pipe_dc_fe = load_pkl_file(
        model_dir / "clf_pipeline_data_cleaning_feat_eng.pkl"
    )

    death_event_pipe_model = load_pkl_file(
        model_dir / "clf_pipeline_model.pkl"
    )

    # -----------------------------------
    # Load training and test datasets
    # -----------------------------------

    X_train = pd.read_csv(model_dir / "X_train.csv")
    X_test = pd.read_csv(model_dir / "X_test.csv")

    y_train = pd.read_csv(
        model_dir / "y_train.csv"
    ).iloc[:, 0].to_numpy()

    y_test = pd.read_csv(
        model_dir / "y_test.csv"
    ).iloc[:, 0].to_numpy()

    # -----------------------------------
    # Model information
    # -----------------------------------

    st.info(
        """
        This supervised machine learning model was developed
        to classify the recorded death-event outcome among
        patients with heart failure.

        Model performance is evaluated using precision,
        recall, F1-score, and the confusion matrix.

        Particular attention is given to recall for
        DEATH_EVENT = 1.
        """
    )

    # -----------------------------------
    # Display ML pipelines
    # -----------------------------------

    st.divider()
    st.subheader("Machine Learning Pipelines")

    st.write("#### 1. Data Cleaning and Feature Engineering")
    st.write(death_event_pipe_dc_fe)

    st.write("#### 2. Classification Model")
    st.write(death_event_pipe_model)

    # -----------------------------------
    # Feature importance
    # -----------------------------------

    st.divider()
    st.subheader("Feature Importance")

    st.write("Features used by the trained model:")
    st.write(X_train.columns.tolist())

    feature_image_path = model_dir / "features_importance.png"

    if feature_image_path.is_file():
        st.image(
            str(feature_image_path),
            caption="Feature importance from the trained model",
            use_container_width=True
        )
    else:
        st.warning("Feature importance image was not found.")

    # -----------------------------------
    # Model performance
    # -----------------------------------

    st.divider()
    st.subheader("Model Performance")

    clf_performance(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pipeline=death_event_pipe_model,
        label_map=["No Death Event", "Death Event"]
    )

    st.caption(
        "This model predicts the recorded DEATH_EVENT label. "
        "Its outputs should not be interpreted as validated "
        "individual mortality risk estimates."
    )

