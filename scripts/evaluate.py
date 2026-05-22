"""CLI to evaluate a saved model (placeholder)."""
import argparse
import joblib
import pandas as pd
from src.training.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target])
    preds = model.predict(X)
    metrics = evaluate_model(y, preds)
    print(metrics)


if __name__ == "__main__":
    main()
