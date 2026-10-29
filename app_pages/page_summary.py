import streamlit as st

def page_summary_body():

    


    st.write("### Quick Project Summary")

# Dataset description
    st.info(
        """
        **Project Dataset**

        The dataset contains clinical records of **299 patients with heart failure**
        collected at the Faisalabad Institute of Cardiology and Allied Hospital
        in Pakistan between April and December 2015.

        It includes demographic and clinical characteristics such as age, sex,
        anaemia, diabetes, high blood pressure, ejection fraction, platelet count,
        serum creatinine, and serum sodium.

        The dataset also includes follow-up duration (`time`) and an outcome
        variable (`DEATH_EVENT`) indicating whether a death was recorded
        during the follow-up period.
    """
)

# Project documentation
    st.write(
        "For additional information, please refer to the "
        "[Project README file](https://github.com/)."
)

# Project objectives
    st.success(
        """
        **Project Objectives**

        This project aims to analyze clinical characteristics and outcomes
        among patients with heart failure using machine learning.

        It includes:

        - **Supervised learning:** Predicting the recorded death-event outcome
        using clinical characteristics.
        - **Unsupervised learning:** Identifying patient subgroups using
        K-Means clustering.
        - **Model interpretation:** Investigating which clinical variables
        contribute to predictions and cluster assignments.
    """
)
