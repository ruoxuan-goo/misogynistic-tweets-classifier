# Misogynistic Tweets Classifier
A prove of concept tool that enables non-ML experts to carry out ML operations and enabling the users to interpret the prediction results easily. The application is built with [Python](https://www.python.org/) and [Streamlit](https://docs.streamlit.io/) and have intergrated two supervised machine learning model.

Supervised machine learning is a process where data is labeled and used to train a model to recognize patterns. This model can then be used to make predictions about outcomes of interest, such as whether if a tweet is misogynistic.

# Proposed Pipeline 
![App pipeline](/pipeline.png)

# Future Roadmaps
Future research can investigate ways to scale the system, including adding more deep-learning models, expanding its application to new domains, and deploying the models via REST APIs so that the front-end can be developed using a more modern front-end framework.

# How to run the application 
1. Clone the project
2. Download the word vector and place it in pages/vectors
3. In terminal, streamlit run Home.py

# Acknowledgement 
I am extremely grateful to my supervisor Dr Md Abul Bashar for providing me guidance and support throughout the research project. I am extremely grateful to my supervisor Associate Professor Richi Nayak for providing me the valuable opportunity to work on this research project. I am extremely grateful for the contributions of the participants who were part of the experiments. 

ML Model:
-	[CNN model](https://github.com/mdabashar/CNN_for_Misogynistic_Tweet_Detection)
-	[LSTM model](https://github.com/mdabashar/Deep-Learning-Algorithms/blob/master/LSTM%20Hate%20Speech%20Detection.ipynb)

Word vector download [link](https://drive.google.com/file/d/1WFwmijBdrtsxqxHclAMet2TMifvj_cYn/view)





