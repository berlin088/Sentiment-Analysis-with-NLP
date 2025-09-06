Sentiment Analysis of Customer Reviews
Overview

This project performs Sentiment Analysis on customer reviews using Natural Language Processing (NLP) techniques. It leverages TF-IDF vectorization and Logistic Regression to classify reviews as positive, negative, or neutral.

The notebook includes:

Data creation and preprocessing

Text cleaning (lowercasing, removing punctuation/stopwords)

TF-IDF feature extraction

Logistic Regression model training and evaluation

Visualization of results (sentiment distribution and confusion matrix)

Predicting sentiment for new reviews

Features

Preprocessing: Converts raw text into clean, structured format.

TF-IDF Vectorization: Converts text into numerical features for ML models.

Logistic Regression: Efficient model for text classification tasks.

Evaluation: Accuracy, confusion matrix, and classification report.

Prediction: Test the model on new customer reviews.

Technologies Used

Python 3.x

pandas, numpy

scikit-learn

nltk (for stopwords)

matplotlib & seaborn (for visualization)

How to Run

Clone the repository:

git clone <repository-url>


Open the Jupyter Notebook:

jupyter notebook


Run each cell sequentially to see preprocessing, model training, and evaluation.

Sample Predictions
predict_sentiment("I really love this product!")  # Output: positive
predict_sentiment("Worst experience ever.")       # Output: negative

Dataset

A small synthetic dataset of customer reviews is included in the notebook.

Can be replaced with a real dataset (like IMDb or Amazon reviews) for larger-scale analysis.

License

This project is open-source and available under the MIT License.
