
import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from sklearn.metrics import silhouette_score

from src.data_management import (
    load_heart_failure_data,
    load_pkl_file
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cluster_distribution_per_variable(df, target):

    df_bar_plot = (
        df.groupby(["Clusters", target])
        .size()
        .reset_index(name="Count")
    )

    df_bar_plot[target] = df_bar_plot[target].astype(str)

    st.subheader(f"Distribution of {target} Across Clusters")

    fig = px.bar(
        df_bar_plot,
        x="Clusters",
        y="Count",
        color=target,
        barmode="group",
        labels={"Count": "Number of Patients"}
    )

    fig.update_xaxes(type="category")

    st.plotly_chart(fig, use_container_width=True)

    # Relative distribution within each cluster
    df_relative = (
        df.groupby(["Clusters", target])
        .size()
        .unstack(fill_value=0)
        .pipe(lambda counts: counts.div(counts.sum(axis=1), axis=0) * 100)
        .stack()
        .reset_index(name="Relative Percentage (%)")
    )

    df_relative[target] = df_relative[target].astype(str)

    st.subheader(f"Percentage of {target} Within Each Cluster")

    fig = px.bar(
        df_relative,
        x="Clusters",
        y="Relative Percentage (%)",
        color=target,
        barmode="group",
        labels={"Relative Percentage (%)": "Percentage (%)"}
    )

    fig.update_xaxes(type="category")

    st.plotly_chart(fig, use_container_width=True)


def page_cluster_body():

    version = "v1"

    model_dir = (
        PROJECT_ROOT
        / "outputs"
        / "ml_pipeline"
        / "cluster_analysis"
        / version
    )

    st.title("ML Pipeline: Heart Failure Cluster Analysis")

    # ---------------------------------
    # Load saved clustering pipeline
    # ---------------------------------
    cluster_pipe = load_pkl_file(
        model_dir / "cluster_pipeline.pkl"
    )

    # ---------------------------------
    # Load original dataset
    # ---------------------------------
    df = load_heart_failure_data().copy()

    # ---------------------------------
    # Load cluster features
    # ---------------------------------
    
    # ---------------------------------
    # Load features used during training
    # ---------------------------------
    if hasattr(cluster_pipe, "feature_names_in_"):
        cluster_features = list(cluster_pipe.feature_names_in_)
    else:
        st.error(
            "The clustering pipeline does not contain feature_names_in_. "
            "Check the features used during model training."
        )
        return

    missing_features = [
        col for col in cluster_features
        if col not in df.columns
    ]

    if missing_features:
        st.error(
            f"Features missing from original dataset: {missing_features}"
        )
        return

    X_cluster = df.loc[:, cluster_features].copy()

    # ---------------------------------
    # Predict cluster labels
    # ---------------------------------
    cluster_labels = cluster_pipe.predict(X_cluster)

    df_cluster = df[["DEATH_EVENT"]].copy()

    df_cluster["Clusters"] = cluster_labels

    # ---------------------------------
    # Show pipeline
    # ---------------------------------
    st.subheader("Clustering Pipeline")
    st.write(cluster_pipe)

    st.subheader("Features Used for Clustering")
    st.write(cluster_features)

    # ---------------------------------
    # Silhouette score
    # ---------------------------------
    st.divider()
    st.subheader("Cluster Evaluation")

    if len(set(cluster_labels)) > 1:

        # Evaluate in the same transformed space
        # used by the K-Means estimator.
        X_transformed = cluster_pipe[:-1].transform(
            X_cluster
        )

        score = silhouette_score(
            X_transformed,
            cluster_labels
        )

        st.metric(
            "Average Silhouette Score",
            f"{score:.3f}"
        )

    # ---------------------------------
    # Silhouette plot
    # ---------------------------------
    silhouette_path = (
        model_dir / "clusters_silhouette.png"
    )

    if silhouette_path.is_file():
        st.image(
            str(silhouette_path),
            caption="Cluster Silhouette Analysis",
            use_container_width=True
        )

    # ---------------------------------
    # Cluster distribution
    # ---------------------------------
    st.divider()

    cluster_distribution_per_variable(
        df=df_cluster,
        target="DEATH_EVENT"
    )

    # ---------------------------------
    # Features defining clusters
    # ---------------------------------
    st.divider()
    st.subheader("Features Defining the Clusters")

    features_path = (
        model_dir / "features_define_cluster.png"
    )

    if features_path.is_file():
        st.image(
            str(features_path),
            use_container_width=True
        )

    # ---------------------------------
    # Cluster profile
    # ---------------------------------
    st.divider()
    st.subheader("Cluster Profile")

    profile_path = model_dir / "clusters_profile.csv"

    if profile_path.is_file():

        cluster_profile = pd.read_csv(profile_path)

        st.dataframe(
            cluster_profile,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "The saved cluster profile was not found."
        )

    # ---------------------------------
    # Outcome summary by cluster
    # ---------------------------------
    st.subheader("Observed Death Event Rate by Cluster")

    outcome_summary = (
        df_cluster.groupby("Clusters")
        .agg(
            Patients=("DEATH_EVENT", "size"),
            Death_Events=("DEATH_EVENT", "sum"),
            Death_Event_Rate=("DEATH_EVENT", "mean")
        )
        .reset_index()
    )

    outcome_summary["Death_Event_Rate"] *= 100

    st.dataframe(
        outcome_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Cluster membership is based on similarity in clinical "
        "features. The observed death-event rates are descriptive "
        "and do not establish clinical risk categories."
    )

