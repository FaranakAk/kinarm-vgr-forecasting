# train_moment_dir3.py
"""
Train a direction-conditioned MOMENT aggregator head and (optionally) generate
MC-dropout forecasts on the test set. This is a cleaned version of
train_ensemble_generate_forecasts_dir3_set2.py with:
- CLI args instead of hard-coded paths
- MOMENT-1-small by default 
- Soft-DTW loss by default (falls back to MSE if Soft-DTW not installed)
- Saves the context+forecast matrix expected by elbow plotting script

Dependencies: existing local modules
  - singletrial_dataset.CustomTimeSeriesDataset_dir3_SingleTrial
  - subject_batch.SubjectBatchSampler, collate_fn_single_subject
  - aggregator_head.SingleTrialAggregatorHead
  - momentfm.MOMENTPipeline
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# local project imports (unchanged)
from singletrial_dataset import CustomTimeSeriesDataset_dir3_SingleTrial
from subject_batch import SubjectBatchSampler, collate_fn_single_subject
from aggregator_head import SingleTrialAggregatorHead

from momentfm import MOMENTPipeline

def _try_softdtw():
    """Return a callable loss() if Soft-DTW is available, else None."""
    try:
        from soft_dtw_cuda import SoftDTW
        return lambda: SoftDTW(gamma=0.01, normalize=True)
    except Exception:
        try:
            from torchsoftdtw import SoftDTW
            return lambda: SoftDTW(gamma=0.01, normalize=True)
        except Exception:
            return None

def build_loss(name: str):
    if name == "softdtw":
        fac = _try_softdtw()
        if fac is not None:
            return fac(), "softdtw"
        print("[warn] Soft-DTW not available, falling back to MSE.")
        return nn.MSELoss(), "mse"
    return nn.MSELoss(), "mse"

def freeze_backbone(pipeline: MOMENTPipeline):
    """Freeze embedder and encoder parameters (paper setup)."""
    for p in pipeline.model.embedder.parameters():
        p.requires_grad = False
    for p in pipeline.model.encoder.parameters():
        p.requires_grad = False

def train_epoch(model, head, loss_fn, loader, device, scaler=None, optimizer=None, grad_clip=None):
    head.train(True)         # train head
    model.eval()             # keep backbone frozen / eval
    losses = []
    for batch in tqdm(loader, desc="train"):
        # batch: (context, forecast, input_mask, context_dirs, forecast_dir, _, _)
        context, forecast, _, context_dirs, forecast_dir, _, _ = batch
        context      = context.to(device)       # (1,8,64) or (C,8,64) -> per-collate is (n_channels,8,64)
        forecast     = forecast.to(device)      # (1,64)
        context_dirs = context_dirs.to(device)  # (8,)
        forecast_dir = forecast_dir.to(device)  # ()

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            out = head(pipeline=model, context=context, context_dirs=context_dirs, forecast_dir=forecast_dir)  # (1,1,64)
            loss = loss_fn(out.squeeze(1), forecast.unsqueeze(0))  # both -> (1,64)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = head(pipeline=model, context=context, context_dirs=context_dirs, forecast_dir=forecast_dir)
                loss = loss_fn(out.squeeze(1), forecast.unsqueeze(0))
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan

@torch.no_grad()
def eval_epoch(model, head, loss_fn, loader, device):
    head.train(False)        # disable dropout for val
    model.eval()
    losses = []
    for batch in loader:
        context, forecast, _, context_dirs, forecast_dir, _, _ = batch
        context      = context.to(device)
        forecast     = forecast.to(device)
        context_dirs = context_dirs.to(device)
        forecast_dir = forecast_dir.to(device)

        out = head(pipeline=model, context=context, context_dirs=context_dirs, forecast_dir=forecast_dir)
        loss = loss_fn(out.squeeze(1), forecast.unsqueeze(0))
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan

def run_training(args, device):
    # dataset + loaders
    train_ds = CustomTimeSeriesDataset_dir3_SingleTrial(data_split="train", channels=args.channels, data_dir=args.data_dir)
    val_ds   = CustomTimeSeriesDataset_dir3_SingleTrial(data_split="val",   channels=args.channels, data_dir=args.data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=SubjectBatchSampler(train_ds, trials_per_subject=9, batch_size=args.batch_size),
        collate_fn=collate_fn_single_subject,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=SubjectBatchSampler(val_ds, trials_per_subject=9, batch_size=args.batch_size),
        collate_fn=collate_fn_single_subject,
    )

    # backbone + head
    ckpt_name = f"AutonLab/MOMENT-1-{args.moment_size}"
    model = MOMENTPipeline.from_pretrained(
        ckpt_name,
        model_kwargs=dict(task_name="forecasting", freeze_encoder=True, freeze_embedder=True,
                          freeze_head=False, forecast_horizon=64),
    ).to(device)
    model.init()
    freeze_backbone(model)
    model.eval()

    head = SingleTrialAggregatorHead(
        d_model=model.config.d_model,
        forecast_horizon=64,
        num_directions=8,
        dropout_prob=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) if args.lr_step>0 else None
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    loss_fn, loss_tag = build_loss(args.loss)

    # training loop
    best_val = float("inf")
    cur_epoch = 1
    train_hist, val_hist = [], []
    os.makedirs(args.results_dir, exist_ok=True)
    best_model_path = os.path.join(args.results_dir, f"MOMENT_best_{args.moment_size}_{loss_tag}.pt")

    while cur_epoch <= args.epochs:
        tr = train_epoch(model, head, loss_fn, train_loader, device, scaler, optimizer, args.grad_clip)
        va = eval_epoch(model, head, loss_fn, val_loader, device)
        train_hist.append(tr); val_hist.append(va)
        print(f"[epoch {cur_epoch:03d}] train={tr:.4f}  val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save({"model_state_dict": model.state_dict(),
                        "head_state_dict": head.state_dict(),
                        "epoch": cur_epoch,
                        "loss": loss_tag,
                        "moment_size": args.moment_size}, best_model_path)
            print(f"  ↳ saved {best_model_path}")

        if scheduler is not None:
            scheduler.step()
        cur_epoch += 1

    # loss curves
    if args.plot:
        plt.figure()
        plt.plot(train_hist, label="train"); plt.plot(val_hist, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, f"loss_{args.moment_size}_{loss_tag}.png"), dpi=180)
        plt.close()

    print(f"[done] best val={best_val:.4f}")
    return best_model_path

@torch.no_grad()
def run_inference(args, device):
    """MC-dropout forecasts with head in train() to keep dropout active."""
    print("Generating forecasts with the single-trial aggregator approach.")
    test_ds = CustomTimeSeriesDataset_dir3_SingleTrial(data_split="test", channels=args.channels, data_dir=args.data_dir)
    test_headers = np.load(os.path.join(args.data_dir, f"vgr_{args.channels}_test_headers.npy"), allow_pickle=True)

    test_loader = DataLoader(
        test_ds,
        batch_sampler=SubjectBatchSampler(test_ds, trials_per_subject=9, batch_size=1),
        collate_fn=collate_fn_single_subject,
        shuffle=False,
    )

    # load backbone + head
    ckpt_name = f"AutonLab/MOMENT-1-{args.moment_size}"
    model = MOMENTPipeline.from_pretrained(
        ckpt_name,
        model_kwargs=dict(task_name="forecasting", freeze_encoder=True, freeze_embedder=True,
                          freeze_head=False, forecast_horizon=64),
    ).to(device)
    model.init(); model.eval()

    head = SingleTrialAggregatorHead(
        d_model=model.config.d_model,
        forecast_horizon=64,
        num_directions=8,
        dropout_prob=args.dropout,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    head.load_state_dict(checkpoint["head_state_dict"], strict=False)

    # keep dropout ON for MC sampling
    head.train(True)

    all_forecasts = []    # list[dict{dir -> list[np.ndarray(64,)]}]
    seen_sessions = set()
    subject_counter = 0

    for batch_data in test_loader:
        context, forecast, input_mask, context_dirs, forecast_dir, _, _ = batch_data
        context      = context.to(device)
        context_dirs = context_dirs.to(device)

        cur_header = test_headers[subject_counter]
        subject_session_id = cur_header[-1].replace(".zip", "")
        subject_counter += 1

        if subject_session_id in seen_sessions:
            continue
        seen_sessions.add(subject_session_id)

        forecasts_for_subject = {}
        for direction_index in range(8):
            fdir = torch.tensor(direction_index, dtype=torch.long, device=device)
            samples = []
            for _ in range(args.num_mc):
                out = head(pipeline=model, context=context, context_dirs=context_dirs, forecast_dir=fdir)  # (1,1,64)
                samples.append(out.squeeze(0).squeeze(0).cpu().numpy())  # (64,)
            forecasts_for_subject[direction_index] = samples
        all_forecasts.append(forecasts_for_subject)

    # Save raw forecasts per subject (list of dicts)
    os.makedirs(args.results_dir, exist_ok=True)
    np.save(os.path.join(args.results_dir, "all_forecasts.npy"), np.array(all_forecasts, dtype=object))
    print(f"[write] {os.path.join(args.results_dir, 'all_forecasts.npy')}")

    # Also save the combined (8*64 + 64*64) matrix expected by plotting code
    ctx_pred_rows = []
    unique_subjects = sorted(set(subj for subj, _ in test_ds.index_list))
    for subject_id in unique_subjects:
        # gather this subject's trials ordered by trial index
        idxs = [i for i, (subj, _) in enumerate(test_ds.index_list) if subj == subject_id]
        if len(idxs) < 9:
            continue
        idxs.sort(key=lambda ix: test_ds.index_list[ix][1])
        # take 8 context trials
        ctx = []
        for ix in idxs:
            item = test_ds[ix]
            if not item["is_forecast"]:
                # item["context"] shape (1,64) or (C,64) depending on dataset, select the single channel
                ctx.append(np.array(item["signal"], dtype=np.float32).reshape(-1))  # expect (64,)
        if len(ctx) != 8:
            continue
        ctx_vec = np.concatenate(ctx, axis=0)  # (512,)

        # forecasts: take MC means for each of the 8 directions, then tile 8-by-8 grid (64 preds total)
        # If you want all MC samples  (8 dirs × num_mc) instead, adapt below.
        subj_dict = next(d for d in all_forecasts if isinstance(d, dict)) if isinstance(all_forecasts, list) else {}
        if not isinstance(subj_dict, dict):
            subj_dict = all_forecasts[0]
        # Build 64 predicted trials: for each dir, take MC mean 8 times to match 64 (or change to desired selection)
        preds64 = []
        for d in range(8):
            mu = np.mean(subj_dict[d], axis=0)  # (64,)
            preds64.extend([mu.copy() for _ in range(8)])
        preds_mat = np.stack(preds64, axis=0)  # (64,64)
        row = np.concatenate([ctx_vec, preds_mat.reshape(-1)], axis=0)  # 512 + 4096 = 4608
        ctx_pred_rows.append(row)

    if ctx_pred_rows:
        mat = np.stack(ctx_pred_rows, axis=0)
        out_path = os.path.join(args.results_dir, "vgr_context_plus_64paths.npy")
        np.save(out_path, mat.astype(np.float32))
        print(f"[write] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    # data/paths
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--channels", default="Vabs")
    # training
    ap.add_argument("--moment-size", choices=["small","large"], default="small")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1, help="subject-batch (9 trials) fits as 1 logical batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", choices=["softdtw","mse"], default="softdtw")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-step", type=int, default=0, help="0 to disable StepLR")
    ap.add_argument("--lr-gamma", type=float, default=0.1)
    ap.add_argument("--plot", action="store_true")
    # inference
    ap.add_argument("--do-infer", action="store_true", help="Run MC-dropout inference after training")
    ap.add_argument("--checkpoint", default="", help="If provided, skip training and use this checkpoint for inference")
    ap.add_argument("--num-mc", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train (unless checkpoint provided)
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = run_training(args, device)

    # Inference (optional)
    if args.do_infer:
        args.checkpoint = ckpt_path
        run_inference(args, device)

if __name__ == "__main__":
    main()
