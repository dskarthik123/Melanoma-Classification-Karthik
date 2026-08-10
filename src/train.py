import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import make_loaders
from src.models import MelanomaViT
from src.metrics import compute

# Column layout produced by MetadataEncoder: age(2), sex(3), site(7)
FIELD_SLICES = {"age": slice(0, 2), "sex": slice(2, 5), "site": slice(5, 12)}


def mask_metadata(meta: torch.Tensor, drop: list) -> torch.Tensor:
    """Zero out ablated fields, keeping input dimensionality fixed so that the
    architecture is identical across ablations."""
    if not drop:
        return meta
    meta = meta.clone()
    for f in drop:
        meta[:, FIELD_SLICES[f]] = 0.0
    return meta


@torch.no_grad()
def evaluate(model, loader, device, use_meta, drop):
    model.eval()
    probs, ys = [], []
    for img, meta, y in loader:
        img = img.to(device, non_blocking=True)
        m = mask_metadata(meta, drop).to(device) if use_meta else None
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(img, m)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def main(a):
    device = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tr, va, meta_dim = make_loaders(
        csv=a.csv, split_csv=a.split_csv, img_dir=a.img_dir, fold_col=a.fold_col,
        batch_size=a.batch_size, balanced=a.balanced, workers=a.workers)

    model = MelanomaViT(meta_dim=meta_dim if a.use_metadata else 0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss()

    out = Path("results") / a.name
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs") / f"{a.name}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    best = -1.0

    for ep in range(a.epochs):
        model.train()
        t0, losses = time.time(), []
        for img, meta, y in tqdm(tr, desc=f"{a.name} ep{ep}", leave=False):
            img, y = img.to(device, non_blocking=True), y.to(device)
            m = mask_metadata(meta, a.drop).to(device) if a.use_metadata else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(img, m), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        y_true, y_prob = evaluate(model, va, device, a.use_metadata, a.drop)
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)),
               "lr": sched.get_last_lr()[0], "secs": round(time.time() - t0, 1),
               **compute(y_true, y_prob)}
        print(json.dumps({k: rec[k] for k in
                          ["epoch", "train_loss", "roc_auc", "pr_auc", "recall", "secs"]}))
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        # Model selection on validation ROC-AUC, applied identically to every
        # experiment. This makes reported validation metrics mildly optimistic;
        # the bias is constant across configurations and is disclosed in the report.
        if rec["roc_auc"] > best:
            best = rec["roc_auc"]
            np.savez(out / "val_preds.npz", y=y_true, p=y_prob, epoch=ep)
            (out / "config.json").write_text(json.dumps(vars(a), default=str, indent=2))

    print(f"[{a.name}] best val ROC-AUC: {best:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, default="splits/patient.csv")
    p.add_argument("--fold-col", default="patient_fold")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--use-metadata", action="store_true")
    p.add_argument("--drop", nargs="*", default=[], choices=list(FIELD_SLICES))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())