
import streamlit as st
import pandas as pd
import numpy as np


def predict_Death_event(
    X_live,
    death_event_features,
    death_event_pipeline_dc_fe,
    death_event_pipeline_model
):
    # Validate required features
    missing_features = [
        col for col in death_event_features
        if col not in X_live.columns
    ]

    if missing_features:
        st.error(f"Missing features: {missing_features}")
        return None

    # Select features in the expected order
    X_live_death_event = X_live.loc[:, death_event_features].copy()

    # Apply fitted cleaning / feature engineering pipeline
    X_processed = death_event_pipeline_dc_fe.transform(
        X_live_death_event
    )

    # Predict outcome
    prediction = death_event_pipeline_model.predict(X_processed)
    predicted_class = int(prediction[0])

    # Get class probabilities
    probabilities = death_event_pipeline_model.predict_proba(X_processed)

    # Find probability of class 1 (DEATH_EVENT)
    classes = death_event_pipeline_model.classes_

    if 1 not in classes:
        st.error("Model does not contain DEATH_EVENT class 1.")
        return None

    death_class_index = list(classes).index(1)
    death_probability = float(probabilities[0, death_class_index])

    # Display results
    st.subheader("Death Event Prediction")

    st.metric(
        label="Model-estimated probability of DEATH_EVENT = 1",
        value=f"{death_probability * 100:.1f}%"
    )

    if predicted_class == 1:
        st.warning("Predicted class: Death Event")
    else:
        st.success("Predicted class: No Death Event")

    st.caption(
        "This is a model-estimated probability for the recorded "
        "DEATH_EVENT label, not a validated individual mortality risk "
        "over a specified time period."
    )

    return predicted_class


def predict_cluster(
    X_live,
    cluster_features,
    cluster_pipeline,
    cluster_profile
):
    # Validate required features
    missing_features = [
        col for col in cluster_features
        if col not in X_live.columns
    ]

    if missing_features:
        st.error(f"Missing features: {missing_features}")
        return None

    # Select clustering features
    X_live_cluster = X_live.loc[:, cluster_features].copy()

    # Predict cluster using the fitted pipeline
    cluster_prediction = cluster_pipeline.predict(X_live_cluster)

    cluster_id = int(cluster_prediction[0])

    st.subheader("Patient Cluster Prediction")

    st.info(
        f"The patient is assigned to **Cluster {cluster_id}**."
    )

    # Display the profile of the assigned cluster
    if cluster_profile is not None:

        profile = cluster_profile.copy()

        # Supports a DataFrame indexed by cluster ID
        # or one containing a "Clusters" column
        if "Clusters" in profile.columns:
            selected_profile = profile[
                profile["Clusters"] == cluster_id
            ]
        else:
            selected_profile = profile.loc[
                profile.index.astype(str) == str(cluster_id)
            ]

        if not selected_profile.empty:
            st.write("#### Cluster Profile")
            st.dataframe(
                selected_profile,
                use_container_width=True
            )
        else:
            st.warning(
                "No profile is available for the assigned cluster."
            )

    st.caption(
        "Cluster membership represents similarity in the selected "
        "clinical features. It is not itself a mortality prediction."
    )

    return cluster_id
