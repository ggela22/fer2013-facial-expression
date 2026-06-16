"""Make a kaggle submission from a saved checkpoint, without retraining anything.

    python -m src.predict --checkpoint outputs/04_vgg_reg/best.pth \
        --test-csv data/test.csv --out outputs/submission.csv

Submission format is `id,emotion`. If you pass a sample submission I'll copy its id column,
otherwise the ids are just 1..N in the row order of test.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from .data import get_test_loader
from .models import build_model
from .utils import NUM_CLASSES


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-csv", default="data/test.csv")
    p.add_argument("--sample-submission", default=None)
    p.add_argument("--out", default="outputs/submission.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]   # the config got saved inside the checkpoint, so rebuild from that
    img_size = cfg["data"].get("img_size", 48)

    model = build_model(cfg["model"]["name"], num_classes=NUM_CLASSES,
                        **cfg["model"].get("params", {})).to(device)
    with torch.no_grad():
        model(torch.zeros(2, 1, img_size, img_size, device=device))  # build the lazy layers first
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader, n = get_test_loader(args.test_csv, batch_size=256, img_size=img_size)
    preds = []
    for x in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)

    ids = np.arange(1, n + 1)
    if args.sample_submission and os.path.exists(args.sample_submission):
        ids = pd.read_csv(args.sample_submission).iloc[:, 0].to_numpy()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"id": ids, "emotion": preds}).to_csv(args.out, index=False)
    print(f"Wrote {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
