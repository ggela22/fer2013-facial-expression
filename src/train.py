"""Train one experiment (described by a yaml config) and log the whole thing to wandb.

Run it from the repo root, e.g.:
    python -m src.train --config configs/04_vgg_reg.yaml --train-csv data/train.csv

Every architecture family gets its own wandb `group` so the dashboard ends up looking like
mlflow (group = experiment, run = one trial). Any cli flag overrides the yaml - that's how
the sweep and the quick one-off runs tweak things without editing files.
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report

import wandb

from .data import (compute_class_weights, denormalize, get_dataloaders,
                   get_test_loader)
from .engine import evaluate, train_one_epoch
from .models import build_model
from . import sanity_checks
from .utils import (EMOTIONS, NUM_CLASSES, count_parameters, save_checkpoint,
                    set_seed)


# ===== config stuff =====
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    """cli flags beat the yaml. used by the quick runs and the sweep."""
    t, d, m = cfg["train"], cfg["data"], cfg["model"]
    if args.epochs is not None:        t["epochs"] = args.epochs
    if args.lr is not None:            t["lr"] = args.lr
    if args.weight_decay is not None:  t["weight_decay"] = args.weight_decay
    if args.optimizer is not None:     t["optimizer"] = args.optimizer
    if args.scheduler is not None:     t["scheduler"] = args.scheduler
    if args.label_smoothing is not None: t["label_smoothing"] = args.label_smoothing
    if args.batch_size is not None:    d["batch_size"] = args.batch_size
    if args.augment is not None:       d["augment"] = args.augment
    if args.dropout is not None:       m.setdefault("params", {})["dropout"] = args.dropout
    if args.seed is not None:          cfg["seed"] = args.seed
    if args.subset is not None:        d["subset"] = args.subset
    if args.tag:                       cfg.setdefault("tags", []).extend(args.tag)
    return cfg


def build_optimizer(tcfg, params):
    name = tcfg["optimizer"].lower()
    lr, wd = tcfg["lr"], tcfg.get("weight_decay", 0.0)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=tcfg.get("momentum", 0.9),
                               weight_decay=wd, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(tcfg, optimizer, epochs):
    # second value tells the loop *when* to step the scheduler (per epoch vs on the val metric)
    s = (tcfg.get("scheduler") or "none").lower()
    if s == "none":
        return None, "none"
    if s == "step":
        return (torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=tcfg.get("step_size", 15), gamma=tcfg.get("gamma", 0.1)),
            "epoch")
    if s == "cosine":
        return (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs),
                "epoch")
    if s == "plateau":
        return (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=tcfg.get("gamma", 0.5),
            patience=tcfg.get("plateau_patience", 3)),
            "plateau")
    raise ValueError(f"Unknown scheduler '{s}'")


# ===== wandb logging helpers =====
def log_sample_predictions(model, loader, device, n=24):
    """log a handful of val images with predicted vs true label so I can eyeball them."""
    model.eval()
    x, y = next(iter(loader))
    with torch.no_grad():
        preds = model(x.to(device)).argmax(1).cpu()
    images = []
    for i in range(min(n, len(x))):
        img = denormalize(x[i]).squeeze(0).numpy()
        caption = f"pred: {EMOTIONS[preds[i].item()]} | true: {EMOTIONS[int(y[i])]}"
        images.append(wandb.Image(img, caption=caption))
    wandb.log({"val/predictions": images})


def log_final_diagnostics(model, val_loader, device):
    """confusion matrix + per-class precision/recall/f1 on the val set."""
    _, _, preds, tgts = evaluate(model, val_loader, nn.CrossEntropyLoss(), device,
                                 return_preds=True)
    wandb.log({"val/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=tgts, preds=preds, class_names=EMOTIONS)})

    report = classification_report(tgts, preds, target_names=EMOTIONS,
                                   output_dict=True, zero_division=0)
    table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
    for cls in EMOTIONS:
        r = report[cls]
        table.add_data(cls, r["precision"], r["recall"], r["f1-score"], r["support"])
    wandb.log({"val/classification_report": table})


# ===== the main training routine =====
def run(cfg, args):
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    img_size = d.get("img_size", 48)

    # ---- data ----
    train_loader, val_loader, train_labels = get_dataloaders(
        args.train_csv, batch_size=d["batch_size"], val_split=d.get("val_split", 0.1),
        augment=d.get("augment", True), img_size=img_size,
        num_workers=d.get("num_workers", 2), seed=cfg.get("seed", 42),
        subset=d.get("subset"),
    )

    # ---- model ----
    model = build_model(m["name"], num_classes=NUM_CLASSES, **m.get("params", {}))
    model = model.to(device)
    # push a dummy batch through first so the LazyLinear layers actually get built,
    # otherwise the param count comes out as 0 and wandb.watch sees nothing
    with torch.no_grad():
        model(torch.zeros(2, 1, img_size, img_size, device=device))
    n_params = count_parameters(model)

    # ---- loss ----
    weight = None
    if t.get("class_weights", False):
        weight = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=t.get("label_smoothing", 0.0))

    # ---- optimizer / scheduler ----
    optimizer = build_optimizer(t, filter(lambda p: p.requires_grad, model.parameters()))
    scheduler, sched_mode = build_scheduler(t, optimizer, t["epochs"])

    amp_enabled = t.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ---- wandb ----
    run_config = copy.deepcopy(cfg)
    run_config["num_parameters"] = n_params
    run_config["device"] = device.type
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=cfg.get("experiment_name", m["name"]),
        group=cfg.get("group", m["name"]),
        job_type="train",
        tags=cfg.get("tags", []),
        notes=cfg.get("notes", ""),
        config=run_config,
        mode="disabled" if args.no_wandb else "online",
    )
    print(f"Model '{m['name']}' | {n_params:,} params | device={device.type}")

    # ---- optional sanity checks (forward/backward) ----
    if args.run_sanity:
        summary = sanity_checks.run_all(model, train_loader, criterion, device)
        wandb.run.summary.update(summary)
        # the overfit-a-batch test wrecks the weights, so build a clean model again
        # (and a fresh optimizer/scheduler) before the real training starts
        model = build_model(m["name"], num_classes=NUM_CLASSES, **m.get("params", {})).to(device)
        with torch.no_grad():
            model(torch.zeros(2, 1, img_size, img_size, device=device))
        optimizer = build_optimizer(t, filter(lambda p: p.requires_grad, model.parameters()))
        scheduler, sched_mode = build_scheduler(t, optimizer, t["epochs"])

    # log gradient/weight histograms for the model we're actually training
    # (this is basically the backward check, but logged the whole way through)
    wandb.watch(model, log="all", log_freq=200)

    # ---- training loop ----
    out_dir = os.path.join(args.output_dir, cfg.get("experiment_name", m["name"]))
    best_path = os.path.join(out_dir, "best.pth")
    best_val_acc, best_epoch, epochs_no_improve = 0.0, -1, 0
    patience = t.get("early_stop_patience")

    for epoch in range(t["epochs"]):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=t.get("grad_clip"), scaler=scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if sched_mode == "epoch":
            scheduler.step()
        elif sched_mode == "plateau":
            scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": val_loss, "val/acc": val_acc,
            "train/lr": lr,
            "gap/acc": tr_acc - val_acc,   # train minus val acc = how much we're overfitting
        })
        print(f"epoch {epoch:3d} | train {tr_loss:.3f}/{tr_acc:.3f} "
              f"| val {val_loss:.3f}/{val_acc:.3f} | lr {lr:.2e}")

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch, epochs_no_improve = val_acc, epoch, 0
            save_checkpoint({"model": model.state_dict(), "config": cfg,
                             "val_acc": val_acc, "epoch": epoch}, best_path)
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience}).")
                break

    # ---- final diagnostics, but on the best checkpoint (not the last epoch) ----
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    wandb.run.summary["best_val_acc"] = best_val_acc
    wandb.run.summary["best_epoch"] = best_epoch
    log_final_diagnostics(model, val_loader, device)
    log_sample_predictions(model, val_loader, device)

    # ---- save the best model as a wandb artifact ----
    if os.path.exists(best_path) and not args.no_wandb:
        artifact = wandb.Artifact(f"{cfg.get('experiment_name', m['name'])}-model",
                                  type="model",
                                  metadata={"val_acc": best_val_acc, "params": n_params})
        artifact.add_file(best_path)
        wandb.log_artifact(artifact)

    # ---- make a kaggle submission too, if a test csv was given ----
    if args.test_csv and os.path.exists(args.test_csv):
        _write_submission(model, args, img_size, device, out_dir)

    print(f"DONE: best val acc = {best_val_acc:.4f} @ epoch {best_epoch}")
    wandb.finish()
    return best_val_acc


@torch.no_grad()
def _write_submission(model, args, img_size, device, out_dir):
    loader, n = get_test_loader(args.test_csv, batch_size=256, img_size=img_size)
    model.eval()
    preds = []
    for x in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    # default ids are just 1..N (row order). use the sample submission's ids if we have it.
    ids = np.arange(1, n + 1)
    if args.sample_submission and os.path.exists(args.sample_submission):
        import pandas as pd
        ids = pd.read_csv(args.sample_submission).iloc[:, 0].to_numpy()
    import pandas as pd
    path = os.path.join(out_dir, "submission.csv")
    pd.DataFrame({"id": ids, "emotion": preds}).to_csv(path, index=False)
    print(f"Wrote submission -> {path}")
    if not args.no_wandb:
        art = wandb.Artifact("submission", type="predictions")
        art.add_file(path)
        wandb.log_artifact(art)


def parse_args():
    p = argparse.ArgumentParser(description="Train a FER2013 model from a YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--train-csv", default="data/train.csv")
    p.add_argument("--test-csv", default=None, help="If set, also writes a submission.")
    p.add_argument("--sample-submission", default=None)
    p.add_argument("--output-dir", default="outputs")
    # wandb
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "fer2013"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--no-wandb", action="store_true")
    # behaviour
    p.add_argument("--run-sanity", action="store_true",
                   help="Run forward/backward sanity checks before training.")
    # overrides. the underscore aliases are so the wandb sweep agent can pass them in too
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float)
    p.add_argument("--optimizer")
    p.add_argument("--scheduler")
    p.add_argument("--label-smoothing", "--label_smoothing", dest="label_smoothing", type=float)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--subset", type=int, help="Use only first N train samples (debug/overfit).")
    p.add_argument("--augment", dest="augment", action="store_true", default=None)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--tag", action="append", default=[])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    run(cfg, args)
