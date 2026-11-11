# Fake Product Review Detection

A machine learning project that classifies product reviews as **original (OR)** or **computer-generated (CG)** using Natural Language Processing (NLP) and a Bidirectional Long Short-Term Memory (BiLSTM) network.

## Project Overview

Online shoppers use product reviews to make purchasing decisions. This project trains a text classification model to recognize patterns associated with computer-generated reviews. The dataset labels are the target used for training; predictions should be treated as model estimates, not proof that a review is fraudulent.

## Dataset

- **Rows:** 40,432 product reviews
- **Labels:** 20,216 CG and 20,216 OR
- **Columns:** `category`, `rating`, `label`, and `text_`
- The dataset used in this notebook is `fake reviews dataset.csv`. It was obtained from the public [Fake Reviews Detection repository](https://github.com/SayamAlt/Fake-Reviews-Detection). The Kaggle competition shown in the project research requires an invitation to join, so this project uses the accessible dataset above.
- `CG` means computer-generated; `OR` means original/human-written, according to the dataset.

## Method

1. Inspect the data, label counts, missing values, categories, and review lengths.
2. Clean review text by lowercasing, removing punctuation and short tokens, and filtering English stop words.
3. Split the data into training and test sets using an 80/20 stratified split (`random_state=42`).
4. Convert text to integer sequences with a Keras tokenizer and pad/truncate each review to 150 tokens.
5. Train a Bidirectional LSTM with an embedding layer, dense layer, dropout, and sigmoid output.
6. Evaluate the model on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix.

## Results

- **Test accuracy:** 91.37%
- **Test loss:** 0.2347
- **Test samples:** 8,087
- **F1-score:** 0.91 for both classes (rounded)

The result is from one stratified train/test split and may vary if the data, preprocessing, random seed, or training settings change.

## Run the Notebook

1. Open `Fake_Product_Review_Detection.ipynb` in Google Colab.
2. Upload `fake reviews dataset.csv` when the notebook asks for it.
3. Run the notebook cells from top to bottom. The training cell takes a few minutes.
4. Read the final classification report and confusion matrix for test performance.

## Project Files

- `Fake_Product_Review_Detection.ipynb` — complete analysis, preprocessing, model training, and evaluation.
- `README.md` — project summary, data details, method, results, and run instructions.

## Limitations

This is an educational classifier trained on one labeled dataset. Its labels describe the dataset's distinction between computer-generated and original reviews; they do not independently verify a reviewer's identity, purchase, or intent. The model may misclassify reviews from other products, platforms, or writing styles.

## Reference

The project adapts the workflow taught in Coursera's NLP Fake News Detector hands-on course (text cleaning, tokenization, padding, recurrent models, and evaluation) to product review classification. The model architecture and task are tailored to this product-review dataset.


## Saved Model Artifacts

`project_artifacts.zip` contains the trained Keras model (`fake_review_bilstm.keras`), the matching tokenizer (`tokenizer.json`), pinned package versions (`requirements.txt`), and evaluation/model settings (`model_metadata.json`). Use these artifacts with the same preprocessing and sequence length used during training. The notebook remains the recommended way to reproduce the full workflow.
