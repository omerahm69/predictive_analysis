
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


def confusion_matrix_and_report(X, y, pipeline, label_map):

    # Make predictions
    predictions = pipeline.predict(X)

    labels = list(range(len(label_map)))

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.write("#### Confusion Matrix")

    cm = confusion_matrix(
        y_true=y,
        y_pred=predictions,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_map,
        yticklabels=label_map,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------
    # Classification Report
    # -----------------------------
    st.write("#### Classification Report")

    report = classification_report(
        y,
        predictions,
        labels=labels,
        target_names=label_map,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    # -----------------------------
    # Accuracy
    # -----------------------------
    accuracy = accuracy_score(y, predictions)

    st.metric(
        label="Accuracy",
        value=f"{accuracy:.2%}"
    )


def clf_performance(
    X_train,
    y_train,
    X_test,
    y_test,
    pipeline,
    label_map
):

    st.subheader("Model Performance")

    st.write("### Training Set")

    confusion_matrix_and_report(
        X_train,
        y_train,
        pipeline,
        label_map
    )

    st.divider()

    st.write("### Test Set")

    confusion_matrix_and_report(
        X_test,
        y_test,
        pipeline,
        label_map
    )

