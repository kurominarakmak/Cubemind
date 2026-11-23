"""Train a Cross next-move policy from a YAML experiment configuration."""

import argparse
from pathlib import Path

import yaml

from cubemind.training.engine import train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    path = train_model(config, args.config)
    print("Saved best checkpoint to {}".format(path))


if __name__ == "__main__":
    main()
