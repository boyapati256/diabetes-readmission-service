"""CLI entry to train a model using `src.training.train`."""
import argparse
import pandas as pd
from src.training.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV input file")
    parser.add_argument("--target", required=True, help="target column name")
    parser.add_argument("--out", required=False, help="path to save model")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target])
    model, acc = train_model(X, y, model_path=args.out)
    print(f"Trained model accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
