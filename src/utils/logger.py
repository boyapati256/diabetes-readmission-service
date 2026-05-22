"""Structured logging helper."""
import logging


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger for simple console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    log = get_logger("diabetes")
    log.info("Logger works")
