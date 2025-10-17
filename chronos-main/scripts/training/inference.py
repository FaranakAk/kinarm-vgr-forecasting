# -*- coding: utf-8 -*-
"""
Created on Tue May  6 21:28:49 2025

@author: fakbarifar
"""
# inference.py
# ----------------
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM
# from gluonts.dataset.arrow import ArrowDataset
import pyarrow.feather as feather

def load_arrow_rows(path):
    """
    Return a list of dicts with 'start' and 'target' just like ArrowDataset.
    """
    table = feather.read_table(path)          # or  pyarrow.ipc.open_file()
    start_col  = table.column("start").to_pylist()
    target_col = table.column("target").to_pylist()
    return [{"start": s, "target": np.asarray(t, dtype=np.float32)}
            for s, t in zip(start_col, target_col)]

from chronos import ChronosConfig, ChronosTokenizer

# ------------------------------------------------------------------
# 1.  paths & config ------------------------------------------------
RUN_DIR          = Path("chronos_runs/run-0")          # adjust if needed
CKPT_DIR         = RUN_DIR / "checkpoint-final"        # best model (after early stop)
TEST_ARROW       = Path("vgr_Vabs_test.arrow")         # test file
CONTEXT_LENGTH   = 512
PRED_LENGTH      = 64
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 2.  load model + tokenizer ---------------------------------------
model  = AutoModelForSeq2SeqLM.from_pretrained(CKPT_DIR).to(DEVICE).eval()
cconf  = ChronosConfig(**model.config.chronos_config)
tokenizer = cconf.create_tokenizer()
# tokenizer = ChronosTokenizer(config=cconf)
# parameters of the uniform‑bin tokenizer
LOW   = cconf.tokenizer_kwargs["low_limit"]   # e.g. −15.0
HIGH  = cconf.tokenizer_kwargs["high_limit"]  # e.g. +15.0
N_BINS          = cconf.n_tokens - cconf.n_special_tokens   # usable bins
SPECIAL_OFFSET  = cconf.n_special_tokens                    # usually 2
BIN_WIDTH       = (HIGH - LOW) / N_BINS
assert cconf.context_length == CONTEXT_LENGTH
assert cconf.prediction_length == PRED_LENGTH

def decode_tokens(token_ids, scale):
    """
    token_ids : 1‑D int array (64)
    scale     : iterable length‑2  [loc, std]

    returns   : 1‑D numpy float array (length 64)
    """
    loc, std = float(scale[0]), float(scale[1])
    token_ids = np.asarray(token_ids, dtype=int)

    # map token id → bin index (skip special tokens)
    idx = token_ids - SPECIAL_OFFSET
    # centre of each bin
    norm_val = LOW + (idx + 0.5) * BIN_WIDTH
    # undo standardisation
    return norm_val * std + loc


# ------------------------------------------------------------------
# 3.  load Arrow rows ----------------------------------------------
# test_ds = ArrowDataset(TEST_ARROW, freq="1ms")   # freq is a dummy label
# rows    = list(test_ds)                          # materialise → list of dicts

rows = load_arrow_rows(TEST_ARROW)


N       = len(rows)
out_mat = np.zeros((N, CONTEXT_LENGTH + PRED_LENGTH), dtype=np.float32)

# ------------------------------------------------------------------
# 4.  helper --------------------------------------------------------
# def forecast_one_series(series: np.ndarray) -> np.ndarray:
#     """Forecast 64 future samples given first 512 context values."""
#     past  = torch.tensor(series[:CONTEXT_LENGTH], dtype=torch.float32).unsqueeze(0)
#     ids, attn, scale = tokenizer.context_input_transform(past)        # B×T
#     with torch.no_grad():
#         pred_ids = model.generate(
#             input_ids     = ids.to(DEVICE),
#             attention_mask= attn.to(DEVICE),
#             max_new_tokens= PRED_LENGTH,
#             do_sample     = True,      # stochastic forecast; set False for greedy
#             top_k         = 50,
#             temperature   = 1.0,
#         )[0, -PRED_LENGTH:]            # take only the newly generated part
#     forecast = tokenizer.decode(pred_ids.cpu(), scale=scale[0])
#     return forecast

# # ------------------------------------------------------------------
# # 5.  loop over subjects -------------------------------------------
# for i, row in enumerate(rows):
#     series = row["target"].astype(np.float32)               # length 576
#     pred   = forecast_one_series(series)
#     out_mat[i] = np.concatenate([series[:CONTEXT_LENGTH], pred])

# # ------------------------------------------------------------------
# # 6.  save result ---------------------------------------------------
# np.save("vgr_context_plus_forecast.npy", out_mat)
# print("saved:", out_mat.shape, "→ vgr_context_plus_forecast.npy")



# ------------------------------------------------------------------
# 4‑bis. helper: 64 sample paths at once ---------------------------
def forecast_64_paths(series: np.ndarray) -> np.ndarray:
    """
    Return (64, 64) array: 64 stochastic forecasts, each length 64.
    """
    past = torch.tensor(series[:CONTEXT_LENGTH], dtype=torch.float32).unsqueeze(0)
    ids, attn, scale = tokenizer.context_input_transform(past)
    loc_std = scale.squeeze().cpu().tolist()      # <- NEW (shape (2,))
    if not isinstance(loc_std, (list, tuple)): # ← scalar safeguard
        loc_std = [loc_std, 1.0]               #   default std = 1.0
    # Generate 64 samples in one call
    with torch.no_grad():
        gen = model.generate(
            input_ids            = ids.to(DEVICE),
            attention_mask       = attn.to(DEVICE),
            max_new_tokens       = PRED_LENGTH,
            do_sample            = True,
            num_return_sequences = 64,       # <<<
            top_k                = 50,
            temperature          = 1.0,
        )                                       # shape (64, 576)
    preds_64 = []
    for row in gen:
        pred_ids = row[-PRED_LENGTH:]          # keep the 64 new tokens
        # preds_64.append(tokenizer.decode(pred_ids.cpu(), scale=scale[0]))
        preds_64.append(decode_tokens(pred_ids.cpu(), scale=loc_std))
    return np.stack(preds_64)                  # (64, 64)

# ------------------------------------------------------------------
# 5‑bis. assemble output matrix ------------------------------------
N = len(rows)
out_mat = np.zeros((N, CONTEXT_LENGTH + 64 * PRED_LENGTH), dtype=np.float32)  # (N, 4608)

for i, row in enumerate(rows):
    series   = row["target"].astype(np.float32)           # length 576
    forecasts = forecast_64_paths(series)                 # (64, 64)
    flat_pred = forecasts.reshape(-1)                     # 4096
    out_mat[i] = np.concatenate([series[:CONTEXT_LENGTH], flat_pred])

# ------------------------------------------------------------------
np.save("vgr_context_plus_64paths.npy", out_mat)
print("saved:", out_mat.shape, "→ vgr_context_plus_64paths.npy")  # (N, 4608)

#%%
# inference
# ----------------------
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM
import pyarrow.feather as feather
from chronos import ChronosConfig

# ------------------------------------------------------------------
# 1. paths & constants ---------------------------------------------
MODEL_ID        = "amazon/chronos-t5-tiny"        # zero‑shot model
TEST_ARROW      = Path("vgr_Vabs_test.arrow")
CONTEXT_LENGTH  = 512
PRED_LENGTH     = 64
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 2. load model & tokenizer ----------------------------------------
model  = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
cconf  = ChronosConfig(**model.config.chronos_config)
tokenizer = cconf.create_tokenizer()              # supplies the transforms

# parameters for manual decode (uniform‑bin)
LOW  = cconf.tokenizer_kwargs["low_limit"]
HIGH = cconf.tokenizer_kwargs["high_limit"]
N_BINS         = cconf.n_tokens - cconf.n_special_tokens
SPECIAL_OFFSET = cconf.n_special_tokens
BIN_WIDTH      = (HIGH - LOW) / N_BINS

def decode_tokens(token_ids, scale):
    """Manual inverse transform from token IDs → real speed values."""
    loc, std = float(scale[0]), float(scale[1])
    idx = np.asarray(token_ids, dtype=int) - SPECIAL_OFFSET
    norm = LOW + (idx + 0.5) * BIN_WIDTH
    return norm * std + loc

# ------------------------------------------------------------------
# 3. load Arrow rows (fallback loader) -----------------------------
def load_arrow_rows(path):
    table = feather.read_table(path)
    return [
        {"start": s, "target": np.asarray(t, dtype=np.float32)}
        for s, t in zip(table["start"].to_pylist(), table["target"].to_pylist())
    ]

rows = load_arrow_rows(TEST_ARROW)
N    = len(rows)

# ------------------------------------------------------------------
# 4. forecast helper (64 sample paths at once) ---------------------
def forecast_64_paths(series: np.ndarray) -> np.ndarray:
    past = torch.tensor(series[:CONTEXT_LENGTH], dtype=torch.float32).unsqueeze(0)
    ids, attn, scale = tokenizer.context_input_transform(past)
    loc_std = scale.squeeze().cpu().tolist()
    if not isinstance(loc_std, (list, tuple)):
        loc_std = [loc_std, 1.0]

    with torch.no_grad():
        # gen = model.generate(
        #     input_ids            = ids.to(DEVICE),
        #     attention_mask       = attn.to(DEVICE),
        #     max_new_tokens       = PRED_LENGTH,
        #     do_sample            = False,
        #     num_beams=64,
        #     num_return_sequences = 64,
        #     top_k                = 50,
        #     temperature          = 1.0,
        # )
        gen = model.generate(
            input_ids     = ids.to(DEVICE),
            attention_mask= attn.to(DEVICE),
            max_new_tokens= PRED_LENGTH,
            do_sample     = True,      # ← match pipeline
            top_p         = 0.95,
            top_k         = None,      # remove top‑k filter
            temperature   = 1.0,
            num_return_sequences = 64,
)


    preds = [
        decode_tokens(row[-PRED_LENGTH:].cpu(), scale=loc_std) for row in gen
    ]
    return np.stack(preds)                          # (64, 64)

# ------------------------------------------------------------------
# 5. assemble output matrix ----------------------------------------
out_mat = np.zeros((N, CONTEXT_LENGTH + 64 * PRED_LENGTH), dtype=np.float32)

for i, row in enumerate(rows):
    series    = row["target"].astype(np.float32)     # length 576
    forecasts = forecast_64_paths(series)            # (64, 64)
    out_mat[i] = np.concatenate(
        [series[:CONTEXT_LENGTH], forecasts.reshape(-1)]
    )

np.save("vgr_context_plus_64paths_0shot_false.npy", out_mat)
print("saved:", out_mat.shape, "→ vgr_context_plus_64paths.npy")


