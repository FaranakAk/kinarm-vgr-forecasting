
"""
Make the Fig.3-style elbow plot with forecast overlays (model-agnostic).
This is a cleaned version of colored.py:
- No os.chdir; paths are CLI args
- Same ICC bootstrap elbow, same overlay squares at x=8
- Works with either MOMENT or Chronos forecast matrices

Inputs:
  --test-dirs         vgr_raw_arrow/vgr_Vabs_test_directions.npy
  --test-lengths      vgr_raw_arrow/vgr_test_orig_lengths.npy
  --test-headers      vgr_raw_arrow/vgr_Vabs_test_headers.npy
  --ctxpred           results/vgr_context_plus_64paths.npy  (N x 4608)
  --control-ref-pkl   control reference pickles used in the script
  --stroke-ref-pkl    (optional if you filter to controls)

Outputs:
  elbow_with_forecasts.png
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- util: load pickle ---
def lp(p):
    with open(p, "rb") as f:
        return pickle.load(f)

# --- ICC(2,1) helper ---
def compute_icc(x, y):
    # x, y: arrays of subject-level scalars (e.g., reference vs. median-of-first-n)
    x = np.asarray(x); y = np.asarray(y)
    assert x.shape == y.shape and x.ndim == 1 and len(x) >= 2
    subj = np.arange(len(x))
    data = np.vstack([x, y]).T  # shape (n,2)
    n = len(subj); k = 2
    subj_means = data.mean(axis=1)
    overall_mean = data.mean()
    BSS = np.sum((subj_means - overall_mean) ** 2)
    BMS = BSS / (n - 1)
    WSS = np.sum((data[:, 0] - subj_means) ** 2 + (data[:, 1] - subj_means) ** 2)
    EMS = WSS / (n * (k - 1))
    return (BMS - EMS) / (BMS + EMS)

# --- reaction time function ---
from vgr_features_2 import calculate_reaction_time_slope

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxpred", required=True, help="Path to vgr_context_plus_64paths.npy (N x 4608).")
    ap.add_argument("--test-dirs", required=True)
    ap.add_argument("--test-lengths", required=True)
    ap.add_argument("--test-headers", required=True)
    ap.add_argument("--control-vabs-pkl", required=True)
    ap.add_argument("--control-headers-pkl", required=True)
    ap.add_argument("--control-lengths-pkl", required=True)
    ap.add_argument("--stroke-vabs-pkl", default=None)
    ap.add_argument("--stroke-headers-pkl", default=None)
    ap.add_argument("--stroke-lengths-pkl", default=None)
    ap.add_argument("--variant", type=int, default=2, choices=[1,2,3], help="VAR flag (task variant index).")
    ap.add_argument("--cohort-flag", default="1.0", help="STROKE_OR_CONTROL flag string used in headers filter.")
    ap.add_argument("--save", default="elbow_with_forecasts.png")
    args = ap.parse_args()

    # ---------- load test meta arrays ---------------
    org_test_dirs    = np.load(args.test_dirs, allow_pickle=True)
    org_test_lengths = np.load(args.test_lengths, allow_pickle=True)
    org_test_headers = np.load(args.test_headers, allow_pickle=True)

    task_variations = {
        '8 target 10cm reach': 64,
        '10cm 4target In&Out RT': 40,
        'reachout_std_(8_target)': 64
    }
    TARGET = list(task_variations.keys())[args.variant - 1]

    # filter test headers by variant + cohort flag 
    tar_ind = [i for i, h in enumerate(org_test_headers)
               if (h[-2] == TARGET and h[-6] == args.cohort_flag)]

    test_dirs    = org_test_dirs[tar_ind]
    test_lengths = org_test_lengths[tar_ind]
    test_headers = org_test_headers[tar_ind]
    N_subj = len(test_headers)

    # ---------- load the context+forecast matrix ----------
    ctx_pred_mat = np.load(args.ctxpred)  # (N, 4608)
    ctx_pred_mat = ctx_pred_mat[:N_subj]  # align to filtered subjects if needed

    context_sig = ctx_pred_mat[:, :512]                             # (N, 512) 8 context trials
    pred_trials = ctx_pred_mat[:, 512:].reshape(N_subj, 64, 64)     # (N, 64, 64)

    # lengths: for each subject use original 8 lengths + per-forecast mean context length
    context_orig_lengths = np.array([np.array(l[:8]).astype(float) for l in test_lengths])
    mean_ctx_len         = np.nanmean(context_orig_lengths, axis=1)

    # ---------- build reference arrays from pickles ----------
    control_Vabs_data = lp(args.control_vabs_pkl)
    control_headers   = lp(args.control_headers_pkl)
    control_lengths   = lp(args.control_lengths_pkl)

    combined_headers = control_headers[:]
    combined_Vabs    = control_Vabs_data[:]
    combined_lengths = control_lengths[:]

    if args.stroke_vabs_pkl and args.stroke_headers_pkl and args.stroke_lengths_pkl:
        stroke_Vabs_data = lp(args.stroke_vabs_pkl)
        stroke_headers   = lp(args.stroke_headers_pkl)
        stroke_lengths   = lp(args.stroke_lengths_pkl)
        combined_headers += stroke_headers
        combined_Vabs    += stroke_Vabs_data
        combined_lengths += stroke_lengths

    matched_samples, matched_lengths, matched_headers = [], [], []
    test_id_col = test_headers[:, -1]
    for h, samp, ln in zip(combined_headers, combined_Vabs, combined_lengths):
        if (h[-1] in test_id_col) and (h[-2] == TARGET):
            matched_samples.append(samp)
            matched_lengths.append(ln)
            matched_headers.append(h)

    # map test header id -> row index
    test_pos = {val: i for i, val in enumerate(test_id_col)}

    # For each subject, compute reference RT (median over all real trials for the matched session)
    control_ref = []
    subject_metrics = []  # list of per-subject lists: RT from first-n recorded trials (for elbow)
    for h, samp, ln in zip(matched_headers, matched_samples, matched_lengths):
        if h[-1] not in test_pos:
            continue
        idx = test_pos[h[-1]]
        # compute RT for every *real* trial in the matched session
        rts = [calculate_reaction_time_slope(sig.astype(np.float32), original_length=l)
               for sig, l in zip(samp, ln)]
        control_ref.append(np.nanmedian(rts))
        # elbow: we will use first-n medians from recorded trials per subject
        subject_metrics.append(rts)

    # ================= elbow (bootstrap median ICC) =================
    np.random.seed(42)
    max_n  = max(len(m) for m in subject_metrics)
    n_boot = 1000
    x_values, median_iccs, lower_cis, upper_cis = [], [], [], []
    subj_indices = np.arange(len(subject_metrics))

    for n in range(1, max_n + 1):
        firstn, refvals = [], []
        for i in subj_indices:
            arr = np.array(subject_metrics[i])
            if len(arr) >= n:
                firstn.append(np.nanmedian(arr[:n]))
                refvals.append(control_ref[i])
        firstn, refvals = np.array(firstn), np.array(refvals)
        if len(refvals) < 2:
            continue
        icc_reps = []
        for _ in range(n_boot):
            bootsamp = np.random.choice(len(refvals), len(refvals), replace=True)
            icc_reps.append(compute_icc(refvals[bootsamp], firstn[bootsamp]))
        median_iccs.append(np.mean(icc_reps))
        lower_cis .append(np.percentile(icc_reps,  2.5))
        upper_cis .append(np.percentile(icc_reps, 97.5))
        x_values.append(n)

    # ================= overlay from forecasts =================
    # choose how many predicted trials to append at the x=8 location
    ALL_PRED_LIST = [[0, 8, 16, 24, 32, 40, 48, 56],
                     [0, 8, 16, 24, 32],
                     [0, 8, 16, 24, 32, 40, 48, 56]]
    PRED_LIST = ALL_PRED_LIST[args.variant - 1]
    COLORS    = ["red", "blue", "green", "cyan", "magenta", "orange", "purple", "brown"]

    overlay_icc = []
    for n_pred in PRED_LIST:
        est_rt = []
        for s in range(N_subj):
            trials  = [context_sig[s, i*64:(i+1)*64] for i in range(8)]
            trials += [pred_trials[s, j] for j in range(n_pred)]
            lengths = list(context_orig_lengths[s]) + [mean_ctx_len[s]] * n_pred
            rts = [calculate_reaction_time_slope(sig.astype(np.float32), original_length=l)
                   for sig, l in zip(trials, lengths)]
            est_rt.append(np.nanmedian(rts))
        overlay_icc.append(compute_icc(np.array(control_ref), np.array(est_rt)))

    # ---------------- plot ----------------
    plt.figure(figsize=(8,6))
    err_low  = np.array(median_iccs) - np.array(lower_cis)
    err_high = np.array(upper_cis)   - np.array(median_iccs)

    plt.errorbar(x_values, median_iccs, yerr=[err_low, err_high],
                 fmt='-o', capsize=5, label='Subject-level ICC (mean ± 95% CI)')

    for icc, col, pred in zip(overlay_icc, COLORS, PRED_LIST):
        label = '8 ctx + forecasted trials' if pred == 0 else None
        plt.scatter(8, icc, marker='s', s=80, color=col, zorder=5, label=label)

    plt.xlabel("Number of trials per subject")
    plt.ylabel("ICC (median reaction time)")
    plt.title("Elbow Plot with Forecasted Trials Overlay")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.save, dpi=300)
    print(f"[plot] {args.save}")

if __name__ == "__main__":
    main()
