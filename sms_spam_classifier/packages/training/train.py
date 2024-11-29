"""
Module for training the SMS Spam Classifier.

This module handles the loading of data, preprocessing, training the
Naive Bayes classifier, and saving the trained model to a file.
"""

import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sms_spam_classifier.packages.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def train_model(data_file: str, output_dir: str) -> None:
    """
    Train the SMS Spam Classifier and save the model.

    Args:
        data_file (str): Path to the dataset file.
        output_dir (str): Directory to save the trained model file.

    Returns:
        None
    """
    logger.info("Starting training process...")

    # Load the dataset from the specified file
    logger.info(f"Loading data from {data_file}")
    data = pd.read_table(data_file, sep="\t", header=None,
                         names=["label", "sms_message"])

    # Map labels to binary values for model compatibility
    data["label"] = data.label.map({"ham": 0, "spam": 1})

    # Split the dataset into training and testing subsets
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["sms_message"], data["label"], random_state=1
    )

    # Initialize CountVectorizer to convert text to numerical representation
    logger.info("Initializing CountVectorizer...")
    vectorizer = CountVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    # Initialize and train the Naive Bayes model
    logger.info("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_transformed, y_train)

    # Evaluate the model on the test set
    logger.info("Evaluating model...")
    predictions = model.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, predictions)

    # Log accuracy and save the model
    logger.info(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    model_filename = os.path.join(
        output_dir, f"sms-spam-classifier-{int(accuracy * 100)}.pth"
    )

    # Save model and vectorizer to a file
    logger.info(f"Saving model to {model_filename}")
    with open(model_filename, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
