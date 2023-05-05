import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Image Classifier"
    )

    st.write("# Overview")

    st.sidebar.success("Select a classifier above.")

    st.markdown(
        """
        This project contains three Image Classification models.
        
        ### Butterfly Species Classifier

        ### Pneumonia Detector

        ### Document Classifier
    """
    )


if __name__ == "__main__":
    run()