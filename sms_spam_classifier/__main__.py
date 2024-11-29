"""
Main entry point for the SMS Spam Classifier.

This script provides command-line functionality to train
or predict using the classifier.
"""

import argparse
from sms_spam_classifier.packages.training.train import train_model
from sms_spam_classifier.packages.prediction.predict import predict_message


def main() -> None:  # -> None means that the function does not return any value.
    """
    Parse command-line arguments and execute the specified command.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="SMS Spam Classifier")

    # Add subcommands for training and prediction
    parser.add_argument(
        "command", choices=["train", "predict"],
        help="Command to execute: 'train' or 'predict'"
    )
    parser.add_argument(
        "--data_file", help="Path to the training data file (for 'train')"
    )
    parser.add_argument(
        "--output_dir", help="Directory to save the trained model (for 'train')"
    )
    parser.add_argument(
        "--sample_sms", help="SMS message to classify (for 'predict')"
    )
    parser.add_argument(
        "--model_path", help="Path to the trained model (for 'predict')"
    )

    args = parser.parse_args()

    if args.command == "train":
        if not args.data_file or not args.output_dir:
            parser.error("The 'train' command requires --data_file and --output_dir.")
        train_model(args.data_file, args.output_dir)

    elif args.command == "predict":
        if not args.sample_sms or not args.model_path:
            parser.error("The 'predict' command requires --sample_sms and --model_path.")
        predict_message(args.sample_sms, args.model_path)


if __name__ == "__main__":
    main()
