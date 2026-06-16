"""Quick checks to run before wasting time on a full training run.

These are the standard "is my network even working" checks from the lectures:

  1. check_initial_loss   - a fresh 7-class net should basically guess uniformly, so the
                            starting cross-entropy should be around ln(7) ~= 1.946. if it's
                            way off, something's wrong (init, the loss, or input scaling).
  2. overfit_single_batch - if the model + optimizer actually work, they should be able to
                            memorize one tiny batch (loss -> ~0, acc -> 1). if they can't,
                            the backward pass / lr / model is broken somewhere.
  3. check_gradient_flow  - after one backward pass every param should get a real gradient
                            (finite + nonzero). catches dead layers, detached graphs, and
                            vanishing/exploding gradients.
"""
import math

import torch

from .utils import NUM_CLASSES


@torch.no_grad()
def check_initial_loss(model, loader, criterion, device):
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    loss = criterion(model(x), y).item()
    expected = math.log(NUM_CLASSES)
    print(f"[forward check] initial loss = {loss:.4f} | expected ~= ln(7) = {expected:.4f}")
    if abs(loss - expected) > 0.7:
        print("  WARNING: initial loss is far from ln(7) -- check init / normalization / loss.")
    return loss, expected


def overfit_single_batch(model, loader, criterion, device, steps=200, lr=1e-3):
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    acc = (out.argmax(1) == y).float().mean().item()
    print(f"[backward check] overfit 1 batch ({len(y)} samples, {steps} steps): "
          f"loss {history[0]:.4f} -> {history[-1]:.4f}, acc = {acc:.3f}")
    if history[-1] > 0.1:
        print("  WARNING: couldn't overfit a single batch -- model/optimizer is probably buggy.")
    return history, acc


def check_gradient_flow(model, loader, criterion, device):
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    print("[gradient check] mean |grad| per parameter:")
    stats = {}
    problems = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"  {name:40s} -> NO GRAD")
            stats[name] = None
            problems += 1
            continue
        g = p.grad.abs().mean().item()
        stats[name] = g
        flag = ""
        if not math.isfinite(g):
            flag = "  <-- NON-FINITE"
            problems += 1
        elif g == 0.0:
            flag = "  <-- ZERO"
            problems += 1
        print(f"  {name:40s} -> {g:.3e}{flag}")
    print(f"[gradient check] {problems} problematic parameter group(s).")
    return stats


def run_all(model, loader, criterion, device, overfit_steps=200):
    """just runs all three checks and returns a little summary dict for wandb."""
    init_loss, expected = check_initial_loss(model, loader, criterion, device)
    history, of_acc = overfit_single_batch(model, loader, criterion, device, steps=overfit_steps)
    check_gradient_flow(model, loader, criterion, device)
    return {
        "sanity/initial_loss": init_loss,
        "sanity/expected_loss": expected,
        "sanity/overfit_final_loss": history[-1],
        "sanity/overfit_acc": of_acc,
    }
