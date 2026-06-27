"""Command-line entry point for the Crop Disease Prediction pipeline.

Examples
--------
    python main.py generate-demo          # create a synthetic demo dataset
    python main.py train                  # train the CNN
    python main.py train --epochs 25      # train for more epochs
    python main.py evaluate               # scikit-learn performance report
    python main.py predict path/to/leaf.jpg
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Crop Disease Prediction pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-demo", help="generate a synthetic demo dataset")

    p_train = sub.add_parser("train", help="train the CNN on data/")
    p_train.add_argument("--epochs", type=int, default=None)

    sub.add_parser("evaluate", help="evaluate the trained model with scikit-learn")

    p_pred = sub.add_parser("predict", help="classify a single leaf image")
    p_pred.add_argument("image", help="path to a leaf image")

    args = parser.parse_args()

    if args.command == "generate-demo":
        from scripts.generate_demo_data import generate

        generate()
    elif args.command == "train":
        from src import config
        from src.train import train

        train(epochs=args.epochs or config.EPOCHS)
    elif args.command == "evaluate":
        from src.evaluate import evaluate

        evaluate()
    elif args.command == "predict":
        from src.predict import predict

        predict(args.image)


if __name__ == "__main__":
    main()
