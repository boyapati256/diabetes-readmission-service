from src.utils.logger import get_logger
from src.training.train import run_training
import sys
import yaml
import argparse
from pathlib import Path

# Make sure src/ is importable when running from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the diabetes readmission model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the training config YAML (default: configs/train.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_id, metrics = run_training(config)

    logger.info("=" * 50)
    logger.info("Training complete.")
    logger.info(f"  MLflow run ID  : {run_id}")
    logger.info(f"  Test ROC-AUC   : {metrics.get('test_roc_auc')}")
    logger.info(f"  Test PR-AUC    : {metrics.get('test_pr_auc')}")
    logger.info(f"  Test Brier     : {metrics.get('test_brier_score')}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
