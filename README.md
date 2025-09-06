Sentiment Analysis of Customer Reviews
🚀 Project Overview

This project performs Sentiment Analysis on customer reviews using Natural Language Processing (NLP) techniques. It leverages TF-IDF vectorization and Logistic Regression to classify reviews into positive, negative, or neutral sentiments.

The project showcases the full pipeline:

Data preprocessing and cleaning

Text vectorization using TF-IDF

Model training with Logistic Regression

Evaluation using accuracy, classification report, and confusion matrix

Prediction on new customer reviews

Visualization of sentiment distribution and model performance

📂 Features

Text Preprocessing: Converts raw text into clean, structured data.

TF-IDF Vectorization: Transforms text into numerical features suitable for ML models.

Logistic Regression: Efficient algorithm for text classification.

Model Evaluation: Accuracy, confusion matrix, and classification report to analyze performance.

Prediction: Test sentiment prediction on new reviews.

🛠 Technologies Used

Python 3.x

pandas & numpy

scikit-learn

nltk (for stopwords removal)

matplotlib & seaborn (for visualization)

📈 How to Run

Clone the repository:

git clone <repository-url>


Navigate to the project folder and open the Jupyter Notebook:

jupyter notebook


Run each cell sequentially to see data preprocessing, model training, evaluation, and predictions.

📝 Sample Predictions
predict_sentiment("I really love this product!")  # Output: positive
predict_sentiment("This is the worst purchase ever.")  # Output: negative
predict_sentiment("It works okay, nothing special.")  # Output: neutral

📊 Dataset

A synthetic dataset of customer reviews is included for demonstration.

Can easily be replaced with a larger, real dataset (e.g., Amazon reviews, IMDb dataset) for practical use.

🔖 License

This project is open-source and available under the MIT License.
