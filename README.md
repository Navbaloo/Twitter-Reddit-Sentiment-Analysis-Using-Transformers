# Twitter and Reddit Sentiment Analysis Using Transformers

This project performs sentiment analysis on Twitter and Reddit data using a pre-trained Transformer model (DistilBERT). The goal is to classify comments and tweets into positive, negative, or neutral sentiment.

## Features

  * **Data Integration**: Combines datasets from both Twitter and Reddit for a comprehensive analysis.
  * **Data Cleaning**: Preprocesses the text data by handling missing values and removing duplicates.
  * **Sentiment Classification**: Utilizes the DistilBERT model from the Hugging Face Transformers library for sentiment analysis.
  * **TensorFlow Integration**: Built with TensorFlow, allowing for efficient training and evaluation of the model.
  * **Data Visualization**: Includes data visualization techniques like word clouds to get insights from the text data.

## Dataset

This project uses two datasets:

  * **Twitter Data**: A CSV file containing tweets with their corresponding sentiment labels.
  * **Reddit Data**: A CSV file containing Reddit comments with their sentiment labels.

The sentiment is categorized as:

  * **1**: Positive
  * **0**: Neutral
  * **-1**: Negative

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Navbaloo/Twitter-Reddit-Sentiment-Analysis-Using-Transformers.git
    ```
2.  Install the required libraries:
    ```bash
    pip install pandas wordcloud seaborn matplotlib scikit-learn tensorflow transformers
    ```

## Usage

1.  Open the `File.ipynb` notebook in a Jupyter environment.
2.  Make sure you have the `Twitter_Data.csv` and `Reddit_Data.csv` files in your Google Drive, or modify the file paths in the notebook to point to their location.
3.  Run the cells in the notebook sequentially to preprocess the data, train the model, and evaluate its performance.

## Model

This project uses the `distilbert-base-uncased` model, a smaller and faster version of BERT, fine-tuned for sequence classification. The model is trained to classify text into three sentiment categories.

## Results

The model is trained and evaluated on the combined dataset. The final test accuracy is printed at the end of the notebook execution.
