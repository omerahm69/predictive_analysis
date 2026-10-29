import streamlit as st

import streamlit as st


class MultiPage:
    """
    Class to manage multiple pages in a Streamlit application.
    """

    def __init__(self, app_name: str) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🩺",
            layout="wide"
        )

    def add_page(self, title: str, func) -> None:
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self) -> None:
        st.title(self.app_name)

        page = st.sidebar.radio(
            "Menu",
            self.pages,
            format_func=lambda page: page["title"]
        )

        page["function"]()
