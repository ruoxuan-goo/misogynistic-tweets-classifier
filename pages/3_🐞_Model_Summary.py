import streamlit as st
import matplotlib.pyplot as plt
import os
from PIL import Image

st.set_page_config(
    page_title="Model Summary",
    page_icon="🐞",
)


def pie_chart(tp, tn, fp, fn):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'TP', 'TN', 'FP', 'FN'
    sizes = [tp, tn, fp, fn]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    text = ['True Positive ' + str(tp), 'True Negative ' + str(tn),
            'False Positive ' + str(fp), 'False Negative ' + str(fn)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', pctdistance=1.25,
            colors=colors, shadow=False, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    ax1.legend(text, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig1)


def read_file(file):
    f = open(file, 'r')
    text = f.readlines()
    f.close()
    return text


txt_file_list = [file for file in os.listdir(
    "./mymodel") if file.endswith(".txt")]
txt_file_names = [os.path.splitext(file)[0] for file in txt_file_list]


def format(data):
    values = []
    for line in data:
        value = line.split(":", 1)[1]
        values.append(value)
    tp = values[0]
    tn = values[1]
    fp = values[2]
    fn = values[3]
    accuracy = values[4]
    precision = values[5]
    recall = values[6]
    f_measure = values[7]
    cohen = values[8]
    auc = values[9]
    roc_auc = values[10]
    average_roc_auc = values[11]

    return tp, tn, fp, fn, accuracy, precision, recall, f_measure, cohen, auc, roc_auc, average_roc_auc


def view_model(num):

    st.subheader("Model " + str(num+1))

    key1 = "model" + str(num)
    key2 = "display" + str(num)

    # Choose a model
    model = st.selectbox("Choose model", txt_file_names, key=key1)
    file_path = "./mymodel/" + str(model) + ".txt"
    # Read txt summary
    data = read_file(file_path)
    tp, tn, fp, fn, accuracy, precision, recall, f_measure, cohen, auc, roc_auc, average_roc_auc = format(
        data)
    # Get accuracy plot
    image_path = "./output/" + str(model) + ".jpg"

    if model:
        display = ['Summary', 'Confusion pie chart',
                   'Training accuracy and loss']
        visualize = st.selectbox("Display", display, key=key2)

        if visualize == 'Summary':
            st.write(data)

        if visualize == 'Confusion pie chart':
            pie_chart(tp, tn, fp, fn)

        if visualize == 'Training accuracy and loss':
            try:
                accuracy_plt = Image.open(image_path)
                st.image(accuracy_plt)
            except:
                st.warning("Nothing to display")


if __name__ == "__main__":
    st.title('Model Summary')
    col1, col2 = st.columns(2)
    with col1:
        num = st.number_input("Display number of model",
                              min_value=1, max_value=3, value=1)
    for x in range(0, num):
        view_model(x)
