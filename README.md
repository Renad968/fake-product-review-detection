# Fake Product Review Detection

An NLP project that uses a Bidirectional LSTM to classify product reviews as computer-generated or original human-written reviews.

## Project Overview

Online shoppers use product reviews to help make purchasing decisions. This project explores how Natural Language Processing (NLP) and deep learning can classify review text into two dataset labels: computer-generated (CG) and original (OR).

## Dataset

The dataset contains product review text, product category, rating, and label.

- `CG`: Computer-generated review
- `OR`: Original human-written review

Dataset source: [Fake Reviews Detection](https://github.com/SayamAlt/Fake-Reviews-Detection)

The dataset labels distinguish computer-generated text from original human-written text. They do not necessarily identify paid, deceptive, or fraudulent reviews.

## Method

1. Inspected and explored the review dataset.
2. Cleaned review text by lowercasing it, removing punctuation and common English stopwords, and filtering short words.
3. Converted labels to numeric values: `CG = 1` and `OR = 0`.
4. Split the data into training and testing sets using an 80/20 split.
5. Tokenized the text and padded each review to 150 tokens.
6. Trained a Bidirectional LSTM classifier.
7. Evaluated the model on the held-out test set.

## Model

The model uses an embedding layer, a Bidirectional LSTM layer, dense layers, and a sigmoid output for binary classification. It was trained with the Adam optimizer and binary cross-entropy loss.

## Results

- Test accuracy: **90.76%**
- F1-score for Original (OR): **0.91**
- F1-score for Computer-Generated (CG): **0.91**

The notebook also reports precision, recall, and a confusion matrix for the test set.

## How to Run

1. Download the dataset CSV from the source repository.
2. Open `Fake_Product_Review_Detection.ipynb` in Google Colab.
3. Run the notebook cells in order.
4. Upload the dataset CSV when prompted.

## Limitations

This model detects computer-generated review text as represented in this dataset. It is not a general-purpose detector of every type of fake, paid, or fraudulent product review. Its performance depends on the dataset and may not generalize to newer AI-generated reviews or other review platforms.

## Attribution

Dataset source: [SayamAlt/Fake-Reviews-Detection](https://github.com/SayamAlt/Fake-Reviews-Detection)

Original dataset paper: Salminen et al., “Creating and Detecting Fake Reviews of Online Products.”
