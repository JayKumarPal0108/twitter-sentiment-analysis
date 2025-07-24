# Twitter-sentiment-analysis
A machine learning model to classify tweets as positive or negative using Python, Scikit-learn, and NLTK.

# Twitter Sentiment Analysis 🐦

This project is a machine learning model that classifies tweets as positive or negative. It uses a Logistic Regression model trained on a dataset of 1.6 million tweets.

## Features
- **Data Preprocessing**: Uses NLTK for text stemming and stopword removal.
- **Feature Extraction**: Converts text into numerical data using `TfidfVectorizer`.
- **Model**: A Logistic Regression model trained with Scikit-learn.
- **Accuracy**: Achieved an accuracy of approximately 77.7% on the test data.

## Dataset
This project uses the "Sentiment140" dataset, which contains 1.6 million tweets extracted using the Twitter API. The data is not included in this repository due to its size. You can download it from Kaggle: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JayKumarPal0108/twitter-sentiment-analysis.git](https://github.com/JayKumarPal0108/twitter-sentiment-analysis.git)
    cd twitter-sentiment-analysis
    ```
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the NLTK stopwords:**
    ```python
    import nltk
    nltk.download('stopwords')
    ```
4.  **Download the dataset** from the link above and place `training.1600000.processed.noemoticon.csv` in the project directory.

5.  Open and run the `Sentiment_Anlaysis.ipynb` notebook in Jupyter or Google Colab.

## Libraries Used
- pandas
- numpy
- scikit-learn
- nltk
- re
