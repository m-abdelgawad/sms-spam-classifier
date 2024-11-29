"""
Module for predicting spam or ham using the SMS Spam Classifier.

This module loads a pre-trained model and makes predictions
on a given SMS message, including confidence percentages.
"""

import pickle
from sms_spam_classifier.packages.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def predict_message(message: str, model_path: str) -> None:
    """
    Predict whether an SMS message is spam or not and display confidence.

    Args:
        message (str): SMS message to classify.
        model_path (str): Path to the trained model file.

    Returns:
        None
    """
    logger.info("Loading trained model...\n")

    # Load the model and vectorizer from the specified file
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
        model = saved_data["model"]
        vectorizer = saved_data["vectorizer"]

    logger.info(f"Classifying message: '{message}'\n")

    # Transform the SMS message to numerical format
    transformed_message = vectorizer.transform([message])

    # Predict the label for the SMS message
    prediction = model.predict(transformed_message)
    probabilities = model.predict_proba(transformed_message)

    # Extract the confidence percentage for the prediction
    spam_confidence = probabilities[0][1] * 100  # Probability of being spam
    ham_confidence = probabilities[0][0] * 100   # Probability of being ham

    # Determine the result and log the confidence
    result = "Spam" if prediction[0] == 1 else "Ham"
    confidence = spam_confidence if prediction[0] == 1 else ham_confidence

    logger.info(f"Prediction result: {result}")
    logger.info(f"Confidence: {confidence:.2f}%")
