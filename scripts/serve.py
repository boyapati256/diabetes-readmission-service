import sys
import yaml
import uvicorn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    host = config["api"]["host"]
    port = config["api"]["port"]

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
