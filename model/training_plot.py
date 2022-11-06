import time
import streamlit as st
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, filename, iteration_per_epoch, plot_acc):
        self.figname = './output/' + filename + '.jpg'
        self.iteration = iteration_per_epoch
        self.plot = plot_acc

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        self.acc_fig = st.empty()

    def on_epoch_begin(self, epoch, logs={}):
        message = "Start epoch {} of training".format(epoch+1)
        st.write(message)
        self.st_t = st.empty()
        self.progress_bar = st.progress(0.0)

    def on_train_batch_end(self, batch, logs=None):
        if batch > 0:
            self.percent = batch/self.iteration
            self.progress_bar.progress(self.percent)

    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        # self.val_losses.append(logs.get('val_loss'))
        # self.val_acc.append(logs.get('val_acc'))
        self.st_t.write(
            "Training acc. {0:.3f}  Loss {1:.3f}".format(logs.get('accuracy'), logs.get('loss')))
        self.progress_bar.progress(1.0)

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            # plt.plot(N, self.val_losses, label="val_loss")
            # plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch {}".format(epoch))
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            if self.plot:
                self.acc_fig.pyplot(plt)
            plt.savefig(self.figname)
            plt.close()
