"""CLI: train the category classifier and write reports + plots."""
import sys

from .classifier import train_full

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    train_full()
