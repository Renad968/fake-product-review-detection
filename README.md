# Fake Product Review Detection

An NLP project that uses a Bidirectional LSTM to classify product reviews as computer-generated or original.

## Project Overview

This project processes product review text and predicts whether a review is computer-generated (CG) or original (OR). The model uses Natural Language Processing techniques and a Bidirectional LSTM neural network.

## Dataset

The dataset contains product review text, product category, rating, and label.

- `CG`: Computer-generated review
- `OR`: Original, human-written review

Dataset source: [Fake Reviews Detection Dataset](https://github.com/SayamAlt/Fake-Reviews-Detection)

This project detects computer-generated text. Its labels do not necessarily identify paid, deceptive, or fraudulent reviews.

## Method

1. Inspect and clean the review text.
2. Convert labels to numeric values.
3. Split the data into training and testing sets.
4. Tokenize the text and pad sequences to 150 tokens.
5. Train a Bidirectional LSTM model.
6. Evaluate the model on held-out test data.

## Results

- Test accuracy: 90.76%
- F1-score for Original (OR): 0.91
- F1-score for Computer-Generated (CG): 0.91

## How to Run

1. Open `Fake_Product_Review_Detection.ipynb` in Google Colab.
2. Upload the dataset CSV when prompted.
3. Run the notebook cells in order.

## Limitations

The model distinguishes computer-generated reviews from original human-written reviews in this dataset. It should not be interpreted as a general detector of all fake or fraudulent reviews.

## Attribution

Dataset source: [SayamAlt/Fake-Reviews-Detection](https://github.com/SayamAlt/Fake-Reviews-Detection)
