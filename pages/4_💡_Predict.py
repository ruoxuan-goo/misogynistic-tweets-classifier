import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from model.preprocessing import CleanTweets

ct = CleanTweets()

st.set_page_config(
    page_title="Predict",
    page_icon="💡",
)


def load_file():
    """Load text from file"""

    uploaded_file = st.file_uploader(
        "Upload Files", type=['csv'], accept_multiple_files=False)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Display textf
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


def download_file(dataframe):
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(dataframe)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='prediction_result.csv',
        mime='text/csv',
    )


def preprocess_data(data):

    # select column to preprocess
    column = select_column(data)

    # select preprossing task
    st.write("Select task")

    # display checkbox
    textfix_check = st.checkbox("Cleans up erroroneus characters")
    ascii_check = st.checkbox("Remove ASCII")
    start_process = st.button("Preprocess and Predict")

    try:
        if start_process:
            if textfix_check:
                data = ct.cleanTweet(data, column)
            if ascii_check:
                data = ct.cleanNonASCII(data, column)

            display_file(data)
    except:
        st.warning("Unable to preprocess. Please select another column or task.")

    return data


def predict_input(input):

    num_words = 100000
    tokenizer = Tokenizer(num_words=num_words)

    # to number
    tokenizer.fit_on_texts(input)
    xinput = tokenizer.texts_to_sequences(input)
    maxlen = max(map(lambda x: len(x), xinput))
    xinput = pad_sequences(xinput, maxlen=maxlen)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(maxlen,), dtype='int32'))

    for layer in loaded_model.layers[1:]:
        model.add(layer)

    result = model.predict(xinput)

    return result


if __name__ == "__main__":

    # Get list of saved h5 models, which will be displayed in option to load.
    h5_file_list = [file for file in os.listdir(
        "./mymodel") if file.endswith(".h5")]
    h5_file_names = [os.path.splitext(file)[0] for file in h5_file_list]

    st.title("Predict")

    model_type = st.selectbox("Choose a model", h5_file_names)
    loaded_model = tf.keras.models.load_model(
        "./mymodel/{}.h5".format(model_type))

    input_type = st.radio(
        "Select an input type",
        ('Text Input', 'Upload File'))

    # Enter a line of text
    if input_type == 'Text Input':
        input = st.text_input('Enter a line of text')
        predict = st.button('Predict')
        if predict:
            input = [input]
            result = predict_input(input)
            st.subheader("Prediction Result")

            st.subheader(result[0])
            if result[0] >= 0.50:
                st.write("High probability that is a misogynistic tweet")
            else:
                st.write("Low probability that is a misogynistic tweet")

    # Upload a csv file
    if input_type == 'Upload File':

        predict_data = load_file()

        if predict_data is not None:
            st.subheader("Preprocessing")
            # Preprocessing
            predict_processed = preprocess_data(predict_data)
            try:
                input = predict_processed['processed_data'].to_numpy()
                file_result = predict_input(input)
                predict_processed['prediction'] = file_result
                st.subheader("Prediction result:")
                st.dataframe(predict_processed)
                download_file(predict_processed)
            except:
                pass
