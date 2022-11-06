
from sklearn.model_selection import train_test_split
import streamlit as st


# Config
st.set_page_config(
    page_title="Home",
    page_icon="😀",
)

if __name__ == "__main__":

    # App title and description
    st.title("Misogynistic Tweet Detector")
    st.write(
        "This is an application that allows you to train your deep learning model with no code and use it to detect misogynistic tweets")
    st.subheader("How to use:")
    st.write(
        "✅ Step 1: Navigate to 'Upload and Preprocess' page. Upload your labelled dataset in csv file.")
    st.write(
        "✅ Step 2: Select data and preprocessing tasks.")
    st.write(
        "✅ Step 3: Navigate to 'Train Model' page. Adjust hyperparameter. Once you are ready click on the 'Train' button to train your model.")
    st.write(
        "✅ Step 4: In 'Model Summary' page, view summary of your trained models and select up to 3 models to compare their performance.")
    st.write(
        "✅ Step 5: In 'Predict' page, select a trained model, and use it for prediction. You can either enter a line of text or upload a csv file.")
