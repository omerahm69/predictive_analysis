
import streamlit as st
from app_pages.multipage import MultiPage

# Import page functions
from app_pages.page_summary import page_summary_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_Death_event import page_predict_Death_event_body
from app_pages.page_predict_Death_event import page_predict_Death_event_body
from app_pages.page_cluster import page_cluster_body


def main():

    st.set_page_config(
        page_title="Heart Failure Predictive Analysis",
        page_icon="🫀",
        layout="wide"
    )

    # Create the multipage application
    app = MultiPage(
        app_name="Heart Failure Predictive Analysis"
    )

    # Register application pages
    app.add_page(
        "Quick Project Summary",
        page_summary_body
    )
    
    app.add_page(
        "ML: Predict Death Event",
        page_predict_Death_event_body
    )

    app.add_page(
        "ML: Cluster Analysis",
        page_cluster_body
    )

    # Run the application
    app.run()


if __name__ == "__main__":
    main()

