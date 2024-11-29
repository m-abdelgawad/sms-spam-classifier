"""
Module for predicting spam or ham using the SMS Spam Classifier.

This module loads a pre-trained model and makes predictions
on a given SMS message, including confidence percentages.
"""

# Import required modules
import pickle  # For loading serialized Python objects (e.g., model and vectorizer)
from sms_spam_classifier.packages.utils.logger import setup_logger  # For custom logging setup

# Initialize the logger for logging the prediction process
logger = setup_logger()


def predict_message(message: str, model_path: str) -> None:
    """
    Predict whether an SMS message is spam or not and display confidence.

    This function performs the following steps:
    1. Loads a pre-trained model and its associated vectorizer from a file.
    2. Transforms the input SMS message into a numerical format using the vectorizer.
    3. Predicts whether the message is spam or ham using the model.
    4. Calculates and logs the confidence percentages for the prediction.

    Args:
        message (str): The SMS message to classify.
        model_path (str): The path to the trained model file (pickle file).

    Returns:
        None
    """
    # Log the start of the model loading process
    logger.info("Loading trained model...\n")

    # Load the saved model and vectorizer from the specified file
    # The file should contain a dictionary with keys "model" and "vectorizer"
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)  # Load serialized data
        model = saved_data["model"]  # Extract the trained model
        vectorizer = saved_data["vectorizer"]  # Extract the trained vectorizer

    # Log the message being classified
    logger.info(f"Classifying message: '{message}'\n")

    # Transform the SMS message into numerical format
    # `vectorizer.transform` converts the message into the same tokenized format as the training data
    transformed_message = vectorizer.transform([message])

    # Predict the label for the SMS message
    # `model.predict` outputs the predicted class (0 for ham, 1 for spam)
    prediction = model.predict(transformed_message)

    # Predict the probabilities for each class (ham and spam)
    # `model.predict_proba` returns an array where:
    # - [0][0]: Probability of being ham
    # - [0][1]: Probability of being spam
    probabilities = model.predict_proba(transformed_message)

    # Extract the confidence percentages for ham and spam
    spam_confidence = probabilities[0][1] * 100  # Convert spam probability to percentage
    ham_confidence = probabilities[0][0] * 100   # Convert ham probability to percentage

    # Determine the classification result based on the predicted class
    # If the predicted class is 1, it is spam; otherwise, it is ham
    result = "Spam" if prediction[0] == 1 else "Ham"

    # Extract the confidence percentage for the predicted result
    confidence = spam_confidence if prediction[0] == 1 else ham_confidence

    # Log the classification result and its confidence percentage
    logger.info(f"Prediction result: {result}")
    logger.info(f"Confidence: {confidence:.2f}%")
