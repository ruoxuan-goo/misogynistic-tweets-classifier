
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from transformers import pipeline
from model.preprocessing import CleanTweets

# Config
st.set_page_config(
    page_title="Upload and Preprocesss",
    page_icon="📄",
)

ct = CleanTweets()


def load_file():
    """Load text from file"""

    uploaded_file = st.file_uploader(
        "Upload Files", type=['csv'], accept_multiple_files=False)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Display text
            with st.expander("See data"):
                display_file(df)

            return df
        except:
            st.warning("File is empty. Please re-upload a file.")


def display_file(data):
    if data is not None:
        return st.dataframe(data)
    else:
        st.warning("There is nothing to display")


def select_column(data):
    # select column
    option = st.selectbox(
        'Select a column',
        (data.columns))
    return option


def preprocess_data(data):

    # select column to preprocess
    column = select_column(data)

    # select preprossing task
    st.write("Select task")

    # display checkbox
    textfix_check = st.checkbox(
        "Cleans up erroroneus characters", help="e.g. from moooom to mom")
    ascii_check = st.checkbox("Remove ASCII", help="e.g. emojis, html symbol")
    start_process = st.button("Preprocess")

    try:
        if start_process:
            if textfix_check:
                data = ct.cleanTweet(data, column)
            if ascii_check:
                data = ct.cleanNonASCII(data, column)

            display_file(data)
            download_file(data)
    except:
        st.warning("Unable to preprocess. Please select another column or task.")

    return data


def download_file(dataframe):
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(dataframe)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
    )


if __name__ == "__main__":
    if 'processed' not in st.session_state:
        st.session_state.processed = None

    st.title("Upload and Preprocess")
    # Load file
    raw_data = load_file()

    if raw_data is not None:
        st.subheader("Preprocessing")

        # Preprocessing
        st.session_state.processed = preprocess_data(raw_data)
