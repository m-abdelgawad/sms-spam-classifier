"""
Module for training the SMS Spam Classifier.

This module handles the loading of data, preprocessing, training the
Naive Bayes classifier, and saving the trained model to a file.
"""

# Import required modules and libraries
import os  # For file and directory operations
import pickle  # For saving Python objects (e.g., model and vectorizer) to a file
from sklearn.feature_extraction.text import CountVectorizer  # For text vectorization
from sklearn.naive_bayes import MultinomialNB  # For Naive Bayes classification
from sklearn.metrics import accuracy_score  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting data into train/test subsets
import pandas as pd  # For handling tabular data
from sms_spam_classifier.packages.utils.logger import setup_logger  # For custom logging setup

# Initialize the logger for tracking the process and providing useful logs
logger = setup_logger()


def train_model(data_file: str, output_dir: str) -> None:
    """
    Train the SMS Spam Classifier and save the model.

    This function performs the following steps:
    1. Loads the dataset from the given file path.
    2. Preprocesses the data, including converting text labels to binary values.
    3. Splits the data into training and testing subsets.
    4. Vectorizes the text messages into a numerical format using CountVectorizer.
    5. Trains a Multinomial Naive Bayes classifier on the training data.
    6. Evaluates the trained model on the test set.
    7. Saves the trained model and the vectorizer to a file for future use.

    Args:
        data_file (str): Path to the dataset file in tab-separated format (e.g., `.tsv`).
        output_dir (str): Directory where the trained model and vectorizer will be saved.

    Returns:
        None
    """
    logger.info("Starting training process...")

    # Load the dataset
    # Dataset is expected to have two columns: "label" and "sms_message", separated by tabs
    logger.info(f"Loading data from {data_file}")
    data = pd.read_table(data_file, sep="\t", header=None,
                         names=["label", "sms_message"])

    # Map text labels ("ham", "spam") to binary values (0, 1)
    # 'ham' -> 0 (non-spam), 'spam' -> 1 (spam)
    data["label"] = data.label.map({"ham": 0, "spam": 1})

    # Split the dataset into training and testing subsets
    # 75% of the data is used for training, and 25% is used for testing
    # `random_state=1` ensures reproducibility of the split
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["sms_message"], data["label"], random_state=1
    )

    # Initialize CountVectorizer for text preprocessing
    # Converts text data into a matrix of token counts
    # `fit_transform` learns the vocabulary and transforms training data into numerical format
    logger.info("Initializing CountVectorizer...")
    vectorizer = CountVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)

    # Transform the test data using the learned vocabulary
    # Ensures that test data uses the same token-to-index mapping as the training data
    X_test_transformed = vectorizer.transform(X_test)

    # Initialize the Naive Bayes classifier
    # MultinomialNB is well-suited for discrete data like word counts
    logger.info("Training Naive Bayes model...")
    model = MultinomialNB()

    # Train the model using the training data
    model.fit(X_train_transformed, y_train)

    # Make predictions on the test set
    logger.info("Evaluating model...")
    predictions = model.predict(X_test_transformed)

    # Evaluate the model's accuracy
    # `accuracy_score` calculates the proportion of correct predictions
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Model trained with accuracy: {accuracy * 100:.2f}%")

    # Prepare the filename for saving the model
    # Includes the accuracy percentage in the filename for easy identification
    model_filename = os.path.join(
        output_dir, f"sms-spam-classifier-{int(accuracy * 100)}.pth"
    )

    # Save the trained model and vectorizer to a file using `pickle`
    # Allows easy reuse of the model and vectorizer without retraining
    logger.info(f"Saving model to {model_filename}")
    with open(model_filename, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
