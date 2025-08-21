# Bank-note-authentication-using-knn
This project implements a Bank Note Authentication System using the K-Nearest Neighbors (KNN) algorithm.
The goal is to classify whether a banknote is genuine or forged based on its extracted statistical features.

#Project Overview
Counterfeit banknotes are a major financial challenge. By leveraging machine learning techniques, we can build a model that automatically identifies forged banknotes with high accuracy.
In this project, the KNN algorithm is applied to the Bank Note Authentication Dataset, which contains features derived from the Wavelet Transform of images of banknotes.

#Dataset
The dataset used is the Bank Note Authentication Dataset from Kaggle.

#Target:
0 → Genuine Note
1 → Forged Note

#Technologies & Libraries
Language: Python
Jupyter Notebook
NumPy, Pandas → Data Handling
Matplotlib, Seaborn → Visualization
Scikit-learn → Machine Learning (KNN)

#Implementation Steps
-Load and explore the dataset
-Perform preprocessing and visualization
-Split dataset into training and testing sets
-Apply KNN classifier
-Evaluate model performance using accuracy, confusion matrix, and classification report

#Results
The KNN model was successfully trained and tested.
Achieved high accuracy 97% in detecting forged vs. genuine notes.
