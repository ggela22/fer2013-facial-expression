"""The actual train/eval loops for one epoch. Nothing fancy in here."""
import numpy as np
import torch

from .utils import AverageMeter


def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip=None, scaler=None):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    amp_enabled = scaler is not None and scaler.is_enabled()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # mixed precision when we're on gpu, otherwise this is basically a no-op
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = model(x)
            loss = criterion(out, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)   # have to unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = y.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update((out.argmax(1) == y).float().mean().item(), bs)

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    preds, tgts = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss_m.update(loss.item(), y.size(0))
        acc_m.update((out.argmax(1) == y).float().mean().item(), y.size(0))
        if return_preds:   # keep predictions around for the confusion matrix later
            preds.append(out.argmax(1).cpu().numpy())
            tgts.append(y.cpu().numpy())

    if return_preds:
        return loss_m.avg, acc_m.avg, np.concatenate(preds), np.concatenate(tgts)
    return loss_m.avg, acc_m.avg
