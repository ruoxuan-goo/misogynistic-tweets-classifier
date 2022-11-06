import time
from gensim.models import KeyedVectors
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from keras.models import Model
from keras.layers import Input, concatenate, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import random as rn
import tensorflow as tf
import os
from keras import backend as K
import re
import html
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import pandas as pd
import nltk
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from model.training_plot import TrainingPlot
# from evaluate_classification import EvaluateBinaryClassification

nltk.download('punkt')

st.set_page_config(
    page_title="Train Model",
    page_icon="👟",
)

SEED = 123

# reference: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(SEED)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.compat.v1.reset_default_graph()
tf.compat.v1.set_random_seed(SEED)

sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

re1 = re.compile(r' +')


def split_data_and_labels(df):
    examples = []
    labels = []
    for i in df.index:
        examples.append(df['processed_data'][i])
        if df['label'][i] == 0:
            labels.append(0)
        else:
            labels.append(1)
    return examples, labels


# Split data to training and test set
def split_train_and_test(df):
    X, y = split_data_and_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    ytrain = np.array(y_train)
    ytest = np.array(y_test)

    num_words = 100000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)
    xtrain = tokenizer.texts_to_sequences(X_train)
    maxlen = max(map(lambda x: len(x), xtrain))
    xtrain = pad_sequences(xtrain, maxlen=maxlen)

    xtest = tokenizer.texts_to_sequences(X_test)
    xtest = pad_sequences(xtest, maxlen=maxlen)

    return num_words, tokenizer, maxlen, xtrain, xtest, ytrain, ytest


if __name__ == "__main__":

    st.title("Train Model")

    # Get processed data from home
    try:
        df = st.session_state.processed
    except:
        st.warning(
            "Cannot find processed data. Please upload and preprocess a dataset")

    # Create two tabs
    cnn_tab, lstm_tab = st.tabs(["CNN Model", "LSTM Model"])

    # CNN
    with cnn_tab:
        # Take user input
        epochsnum = st.slider("Epoch", min_value=1,
                              max_value=80, value=10, step=1, key="cnn_epoch_slider")
        batch = st.slider(
            "Batch - best to increase or decrease by two folds", min_value=8, max_value=256, value=32, key="cnn_batch_slider")
        activationFunc = st.selectbox(
            'Dense activation function',
            ('sigmoid', 'relu', 'linear'), key="cnn_activation_select")

        loss_func = st.selectbox(
            'Loss function',
            ('binary_crossentropy', 'categorical_crossentropy'), key="cnn_loss_select")

        optimizer = st.selectbox(
            'Optimizer',
            ('adam', 'SGD'), key="cnn_optimizer_select")
        filename = st.text_input('Save your model as',
                                 'model_name', key="cnn_filename")
        plot_acc = st.checkbox(
            'Show real time training loss and accuracy', key="cnn_checkbox", help="Only shows after two epochs")
        train = st.button('Train', key="cnn_train_button")

        # Click train button
        if train:

            st.write("Preparing for training...")

            st.write("Split dataset to train and test...")
            num_words, tokenizer, maxlen, xtrain, xtest, ytrain, ytest = split_train_and_test(
                df)

            # Load word embeddings
            model_ug_cbow = KeyedVectors.load(
                'C:/Users/User/Desktop/cnn-interface/pages/vectors/vectors.txt')

            embeddings_index = {}
            for w in model_ug_cbow.wv.key_to_index.keys():
                embeddings_index[w] = model_ug_cbow.wv[w]

            embedding_matrix = np.zeros((num_words, 200))
            for word, i in tokenizer.word_index.items():
                if i >= num_words:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            def create_cnn_model():
                tweet_input = Input(shape=(maxlen,), dtype='int32')
                tweet_encoder = Embedding(num_words, 200, weights=[
                    embedding_matrix], input_length=maxlen, trainable=True)(tweet_input)
                tweet_encoder = Dropout(0.5)(tweet_encoder)

                bigram_branch = Conv1D(filters=128, kernel_size=3, padding='valid',
                                       activation='relu', strides=1)(tweet_encoder)
                bigram_branch = GlobalMaxPooling1D()(bigram_branch)
                bigram_branch = Dropout(0.5)(bigram_branch)

                trigram_branch = Conv1D(filters=256, kernel_size=4, padding='valid',
                                        activation='relu', strides=1)(tweet_encoder)
                trigram_branch = GlobalMaxPooling1D()(trigram_branch)
                trigram_branch = Dropout(0.2)(trigram_branch)

                fourgram_branch = Conv1D(
                    filters=512, kernel_size=5, padding='valid', activation='relu', strides=1)(tweet_encoder)
                fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
                fourgram_branch = Dropout(0.2)(fourgram_branch)

                merged = concatenate(
                    [bigram_branch, trigram_branch, fourgram_branch], axis=1)

                merged = Dense(256, activation=activationFunc)(merged)
                merged = Dropout(0.5)(merged)

                merged = Dense(1)(merged)
                output = Activation(activationFunc)(merged)

                model = Model(inputs=[tweet_input], outputs=[output])
                model.compile(loss=loss_func,
                              optimizer=optimizer, metrics=['accuracy'])
                return model

            # Calculate total iteration per epoch
            iteration_per_epoch = len(xtrain)/batch
            # Init TrainingPlot class for custom callback
            plot_epoch = TrainingPlot(filename, iteration_per_epoch, plot_acc)

            # Create the model
            cnn_model = create_cnn_model()
            st.write("Training the model.....")

            start = time.time()
            # Train the model
            cnn_model.fit(xtrain, ytrain, epochs=epochsnum,
                          batch_size=batch, verbose=1, callbacks=[plot_epoch])
            stop = time.time()
            duration = stop-start
            hrs = int(duration/3600)
            mins = int((duration-hrs*3600)/60)
            secs = duration-hrs*3600-mins*60
            # Display training time
            msg = 'Training took {0} hours {1} minutes and {2:6.2f} seconds'.format(
                hrs, mins, secs)
            st.write(msg)

            # Save model as filename.h5
            cnn_model.save("./mymodel/" + filename + ".h5")

            # Predict
            st.write("Predicting.....")
            p = cnn_model.predict(xtest, verbose=1)
            predicted = [int(round(x[0])) for x in p]
            predicted = np.array(predicted)
            actual = ytest

            # Celebrate
            st.balloons()

            # Display results
            tp = np.count_nonzero(predicted * actual)
            tn = np.count_nonzero((predicted - 1) * (actual - 1))
            fp = np.count_nonzero(predicted * (actual - 1))
            fn = np.count_nonzero((predicted - 1) * actual)

            accuracy = (tp + tn) / (tp + fp + fn + tn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            fmeasure = (2 * precision * recall) / (precision + recall)
            cohen_kappa_score = cohen_kappa_score(predicted, actual)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(
                actual, predicted)
            auc_val = auc(false_positive_rate, true_positive_rate)
            roc_auc_val = roc_auc_score(actual, predicted)

            st.subheader("Model summary ")
            st.write('True Positive', tp)
            st.write('True Negative', tn)
            st.write('False Positive', fp)
            st.write('False Negative', fn)
            st.write('Accuracy', accuracy)
            st.write('Precision', precision)
            st.write('Recall', recall)
            st.write('f-measure', fmeasure)
            st.write('cohen_kappa_score', cohen_kappa_score)
            st.write('auc', auc_val)
            st.write('roc_auc', roc_auc_val)
            st.write("Average of ROC-AUC score: %.3f" %
                     roc_auc_score(ytest, p))

            # Write to txt file, create new file if file does not exist
            lines = ["True Positive: " + str(tp) + "\n", "True Negative: " + str(tn) + "\n", "False Positive: " + str(
                fp) + "\n", "False Negative: " + str(fn) + "\n", "Accuracy: " + str(accuracy) + "\n", "Precision: " + str(precision) + "\n",
                "Recall: " + str(recall) + "\n", "f-measure: " + str(fmeasure) +
                "\n", "cohen_kappa_score: " + str(cohen_kappa_score) + "\n",
                "auc: " + str(auc_val) + "\n", "roc_auc: " + str(roc_auc_val) + "\n", str("Average of ROC-AUC score: %.3f" % roc_auc_score(ytest, p))]
            f = open("./mymodel/" + filename + ".txt", "w")
            f.writelines(lines)
            f.close()

    # LSTM
    with lstm_tab:
        # Take user input
        lstm_epochsnum = st.slider("Epoch", min_value=1,
                                   max_value=80, value=10, step=1, key="lstm_epoch_slider")
        lstm_batch = st.slider(
            "Batch - best to increase or decrease by two folds", min_value=8, max_value=256, value=32, key="lstm_batch_slider")
        lstm_activation = st.selectbox(
            'Output activation function',
            ('sigmoid', 'softmax', 'relu', 'linear'), key="lstm_activation_select")

        lstm_loss_func = st.selectbox(
            'Loss function',
            ('binary_crossentropy', 'categorical_crossentropy'), key="lstm_loss_select")

        lstm_optimizer = st.selectbox(
            'Optimizer',
            ('adam', 'SGD'), key="lstm_optimizer_select")
        filename = st.text_input('Save your model as',
                                 'model_name', key="lstm_filename")
        plot_acc = st.checkbox(
            'Show real time training loss and accuracy', key="lstm_checkbox", help="Only shows after two epochs")
        train = st.button('Train', key="lstm_train_button")

        # Click train button
        if train:

            st.write("Preparing for training...")

            st.write("Split dataset to train and test...")

            # TEST CODE
            # BASE = 'C:/Users/User/Downloads/RUOXUAN/QMI/'
            # fins_train = [BASE + 'train.csv']
            # fins_test = [BASE + 'test.csv']
            # track = 0

            # # We apply only this preprocessing because our data is already preprocessed

            # def cleanNonAscii(text):
            #     '''
            #     Remove Non ASCII characters from the dataset.
            #     Arguments:
            #         text: str
            #     returns:
            #         text: str
            #     '''
            #     return ''.join(i for i in text if ord(i) < 128)

            # df_train = pd.read_csv(fins_train[track])
            # st.write(df_train)
            # df_train['text'] = df_train['text'].apply(cleanNonAscii)
            # X_train, y_train = df_train['text'].values, df_train['label'].values
            # df_test = pd.read_csv(fins_test[track])
            # st.write(df_test)
            # df_test['text'] = df_test['text'].apply(cleanNonAscii)
            # X_test, y_test = df_test['text'].values, df_test['label'].values
            # num_words = 100000
            # tokenizer = Tokenizer(num_words=num_words)
            # tokenizer.fit_on_texts(X_train)
            # xtrain = tokenizer.texts_to_sequences(X_train)
            # maxlen = max(map(lambda x: len(x), xtrain))
            # xtrain = pad_sequences(xtrain, maxlen=maxlen)

            # xtest = tokenizer.texts_to_sequences(X_test)
            # xtest = pad_sequences(xtest, maxlen=maxlen)
            # TEST CODE END
            num_words, tokenizer, maxlen, xtrain, xtest, ytrain, ytest = split_train_and_test(
                df)

            # Load word embeddings
            model_ug_cbow = KeyedVectors.load(
                'C:/Users/User/Desktop/cnn-interface/pages/vectors/vectors.txt')

            embeddings_index = {}
            for w in model_ug_cbow.wv.key_to_index.keys():
                embeddings_index[w] = model_ug_cbow.wv[w]

            embedding_matrix = np.zeros((num_words, 200))
            for word, i in tokenizer.word_index.items():
                if i >= num_words:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            def create_lstm_model():
                tweet_input = Input(shape=(maxlen,), dtype='int32')
                #tweet_encoder = Embedding(num_words, 200, weights=[embedding_matrix], input_length=maxlen, trainable=True)(tweet_input)
                tweet_encoder = Embedding(
                    num_words, 200, input_length=maxlen)(tweet_input)
                tweet_encoder = Dropout(0.5)(tweet_encoder)
                merged = LSTM(100)(tweet_encoder)
                merged = Dropout(0.5)(merged)
                merged = Dense(1)(merged)
                output = Activation(lstm_activation)(merged)
                model = Model(inputs=[tweet_input], outputs=[output])
                model.compile(loss=lstm_loss_func,
                              optimizer=lstm_optimizer, metrics=['accuracy'])
                # model.summary()
                return model

            # Calculate total iteration per epoch
            iteration_per_epoch = len(xtrain)/batch
            # Init TrainingPlot class for custom callback
            plot_epoch = TrainingPlot(filename, iteration_per_epoch, plot_acc)

            if 'lstm_model' not in st.session_state:
                st.session_state.lstm_model = None

            # Create the model
            st.session_state.lstm_model = create_lstm_model()
            st.write("Training the model.....")

            start = time.time()
            # Train the model
            st.session_state.lstm_model.fit(xtrain, ytrain, epochs=lstm_epochsnum,
                                            batch_size=lstm_batch, verbose=1, callbacks=[plot_epoch])
            stop = time.time()
            duration = stop-start
            hrs = int(duration/3600)
            mins = int((duration-hrs*3600)/60)
            secs = duration-hrs*3600-mins*60
            # Display training time
            msg = 'Training took {0} hours {1} minutes and {2:6.2f} seconds'.format(
                hrs, mins, secs)
            st.write(msg)

            # Save model as filename.h5
            st.session_state.lstm_model.save("./mymodel/" + filename + ".h5")

            # Predict
            st.write("Predicting.....")
            p = st.session_state.lstm_model.predict(xtest, verbose=1)
            predicted = [int(round(x[0])) for x in p]
            predicted = np.array(predicted)
            actual = ytest

            # ebc = EvaluateBinaryClassification(
            #     gnd_truths=actual, predictions=predicted)
            # print(ebc.get_full_report())

            # Celebrate
            st.balloons()

            # Display results
            tp = np.count_nonzero(predicted * actual)
            tn = np.count_nonzero((predicted - 1) * (actual - 1))
            fp = np.count_nonzero(predicted * (actual - 1))
            fn = np.count_nonzero((predicted - 1) * actual)

            accuracy = (tp + tn) / (tp + fp + fn + tn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            fmeasure = (2 * precision * recall) / (precision + recall)
            cohen_kappa_score = cohen_kappa_score(predicted, actual)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(
                actual, predicted)
            auc_val = auc(false_positive_rate, true_positive_rate)
            roc_auc_val = roc_auc_score(actual, predicted)

            st.subheader("Model summary ")
            st.write('True Positive', tp)
            st.write('True Negative', tn)
            st.write('False Positive', fp)
            st.write('False Negative', fn)
            st.write('Accuracy', accuracy)
            st.write('Precision', precision)
            st.write('Recall', recall)
            st.write('f-measure', fmeasure)
            st.write('cohen_kappa_score', cohen_kappa_score)
            st.write('auc', auc_val)
            st.write('roc_auc', roc_auc_val)
            st.write("Average of ROC-AUC score: %.3f" %
                     roc_auc_score(ytest, p))

            # Write to txt file, create new file if file does not exist
            lines = ["True Positive: " + str(tp) + "\n", "True Negative: " + str(tn) + "\n", "False Positive: " + str(
                fp) + "\n", "False Negative: " + str(fn) + "\n", "Accuracy: " + str(accuracy) + "\n", "Precision: " + str(precision) + "\n",
                "Recall: " + str(recall) + "\n", "f-measure: " + str(fmeasure) +
                "\n", "cohen_kappa_score: " + str(cohen_kappa_score) + "\n",
                "auc: " + str(auc_val) + "\n", "roc_auc: " + str(roc_auc_val) + "\n", str("Average of ROC-AUC score: %.3f" % roc_auc_score(ytest, p))]
            f = open("./mymodel/" + filename + ".txt", "w")
            f.writelines(lines)
            f.close()
