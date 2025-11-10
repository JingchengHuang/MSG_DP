"""
DP training script (Temporal Transformer on z-space)

- Reads raw CSV/TXT files under DATA_DIR (same format as AE training).
- For each file: sliding windows of seq_len=10 are encoded by the frozen encoder -> produce z sequence.
- For each file, from the z sequence create samples:
    input: z[s], z[s+1], ..., z[s+k-1]  (k=3)
    cond: emotion one-hot (7) + level scalar (1)  -> total cond_dim=8
    target: z[s+k]
- Train Temporal Transformer to map (k tokens) -> next z.
"""

import os
import time
import datetime
from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------- CONFIG -------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "1030data_sy"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = BASE_DIR / "model" / f"{timestamp}_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# data/encoding params (must match AE training)
SEQ_LEN = 10         # frames per window for encoder -> one z
FEATURE_DIM = 32     # raw control dim
Z_DIM = 64           # latent dim from encoder
K = 3                # number of historical z tokens used as input to DP
COND_DIM = 8         # 7-d one-hot emotion + 1-d level

# DP model / training hyperparams (experience-based; you can tune later)
MODEL_DIM = 128
NHEAD = 8
NUM_TRANSFORMER_LAYERS = 3
FF_DIM = 256
DROPOUT = 0.1

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 500

NUM_WORKERS = 4      # DataLoader workers (Windows -> will be used; script guarded by __main__)
PIN_MEMORY = True

# Paths to previously trained encoder (we expect encoder.pt exists)
ENCODER_PATH = MODEL_DIR / "encoder.pt"

# ------------------- Encoder class (must match saved encoder architecture) -------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_size=128, num_layers=2, z_dim=Z_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, z_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers * num_directions, batch, hidden_size)
        # take last forward+backward states of last layer
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_size*2)
        z = self.fc(h_last)  # (batch, z_dim)
        return z

# ------------------- Temporal Transformer Model -------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, z_dim=Z_DIM, cond_dim=COND_DIM, model_dim=MODEL_DIM,
                 nhead=NHEAD, num_layers=NUM_TRANSFORMER_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_dim = z_dim + cond_dim
        self.model_dim = model_dim
        self.input_fc = nn.Linear(self.input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim, max_len=K+5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(model_dim, z_dim)

    def forward(self, z_seq_cond):
        # z_seq_cond: (batch, k, z_dim + cond_dim)
        # Transformer expects (seq_len, batch, model_dim) or batch_first=False
        x = self.input_fc(z_seq_cond)  # (batch, k, model_dim)
        x = x.permute(1, 0, 2)         # (k, batch, model_dim)
        x = self.pos_enc(x)
        x = self.transformer(x)        # (k, batch, model_dim)
        last = x[-1]                   # (batch, model_dim) -- last time token representation
        z_pred = self.out_fc(last)     # (batch, z_dim)
        return z_pred

# ------------------- Utilities: parse filename for emotion & level -------------------
# Expect filename like "emotion_merged_level_number.csv" or .txt
def parse_emotion_level_from_name(fname):
    """
    解析文件名中的情感类别与强度等级。
    只处理 .csv 文件，若无法解析出 emotion 或 level，则跳过该文件。
    """
    # 仅允许 csv 文件
    if not fname.endswith(".csv"):
        return None, None

    base = os.path.basename(fname)
    name = os.path.splitext(base)[0]  # 去掉扩展名
    parts = name.split('_')

    # 预定义情绪类别
    emotions = ['happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral']
    emotion_index = None
    level_val = None

    # 尝试直接匹配
    for p in parts:
        if p in emotions:
            emotion_index = emotions.index(p)
        if p in ('h', 'l'):
            level_val = 1.0 if p == 'h' else 0.0

    # 如果没匹配到情绪名，再用正则尝试一次
    if emotion_index is None:
        for i, e in enumerate(emotions):
            if re.search(r'\b' + re.escape(e) + r'\b', name):
                emotion_index = i
                break

    # 如果都没找到，则放弃该文件
    if emotion_index is None or level_val is None:
        print(f"[SKIP] 无法解析文件名中的情感或等级: {fname}")
        return None, None

    return emotion_index, float(level_val)

# ------------------- Preprocessing: make z sequences using encoder -------------------
def build_z_sequences(encoder, device):
    """
    Reads all csv/txt files, creates sliding windows of size SEQ_LEN,
    runs encoder in batches to produce z for each window,
    and returns per-file lists so we can make samples z[s..s+k] -> z[s+k].
    """
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    if len(files) == 0:
        raise RuntimeError(f"No csv/txt files found in {DATA_DIR}")

    all_z_per_file = []
    all_emotion_idx_per_file = []
    all_level_per_file = []
    total_windows = 0

    encoder.eval()
    with torch.no_grad():
        for fname in files:
            fpath = DATA_DIR / fname
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                # try with alternative separators
                df = pd.read_csv(fpath, sep=None, engine='python')

            values = df.values[:, :FEATURE_DIM]
            num_frames = values.shape[0]
            num_windows = num_frames - SEQ_LEN

            # ---- 新增安全检查 ----
            if num_windows <= 0:
                print(f"\n[ERROR] File '{fname}' has too few frames ({num_frames}) for SEQ_LEN={SEQ_LEN}.")
                print("       Each file must have more rows than SEQ_LEN to form valid windows.")
                print("       Please check this file’s data integrity.")
                raise ValueError(f"Invalid frame count in file: {fname}")
            # ---------------------

            try:
                # build windows for this file
                windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=(SEQ_LEN, FEATURE_DIM))[0]
            except ValueError as e:
                print(f"\n[ERROR] sliding_window_view() failed for file: {fname}")
                print(f"       File shape: {values.shape}, window_shape: ({SEQ_LEN}, {FEATURE_DIM})")
                print(f"       Error message: {str(e)}")
                raise e  # re-raise to terminate program immediately

            # fallback/manual check
            if windows.shape[0] != num_windows:
                tmp = []
                for s in range(num_windows):
                    tmp.append(values[s:s + SEQ_LEN])
                windows = np.stack(tmp, axis=0)

            # encode
            z_list = []
            BATCH_ENC = 256
            for i0 in range(0, num_windows, BATCH_ENC):
                batch_windows = windows[i0:i0 + BATCH_ENC].astype(np.float32)
                batch_tensor = torch.from_numpy(batch_windows).to(device)
                z_batch = encoder(batch_tensor)
                z_list.append(z_batch.cpu().numpy())
            z_arr = np.concatenate(z_list, axis=0)

            emo_idx, lvl = parse_emotion_level_from_name(fname)
            if emo_idx is None or lvl is None:
                continue

            all_z_per_file.append(z_arr)
            all_emotion_idx_per_file.append(int(emo_idx))
            all_level_per_file.append(float(lvl))
            total_windows += num_windows

    print(f"Preprocessing done. Files processed: {len(all_z_per_file)}, total windows (z vectors): {total_windows}")
    return all_z_per_file, all_emotion_idx_per_file, all_level_per_file


# ------------------- Build DP dataset from z sequences -------------------
class DPZDataset(Dataset):
    def __init__(self, all_z_per_file, all_emotion_idx_per_file, all_level_per_file, k=K):
        """
        all_z_per_file: list of arrays (num_windows, Z_DIM)
        all_emotion_idx_per_file: list of ints
        all_level_per_file: list of floats
        We produce samples where for each file with n windows:
           for s in [0 .. n - (k+1)]:
               input_z = z[s:s+k]    -> shape (k, Z_DIM)
               cond = one-hot(7) + level (1,) -> shape (8,)
               target_z = z[s+k]     -> shape (Z_DIM,)
        """
        self.k = k
        self.z_dim = Z_DIM
        self.cond_dim = COND_DIM
        self.inputs = []   # will store np arrays
        self.targets = []
        self.conds = []

        for z_arr, emo_idx, lvl in zip(all_z_per_file, all_emotion_idx_per_file, all_level_per_file):
            n = z_arr.shape[0]
            max_s = n - (k + 1) + 1  # n - (k+1) + 1 == n - k
            # iterate s = 0 .. n - (k+1)
            if max_s <= 0:
                continue
            # prepare condition vector for entire file (same for all windows)
            onehot = np.zeros(7, dtype=np.float32)
            onehot[emo_idx] = 1.0
            level_arr = np.array([lvl], dtype=np.float32)
            cond_vec = np.concatenate([onehot, level_arr], axis=0)  # shape (8,)

            for s in range(0, n - (k + 1) + 1):
                inp = z_arr[s:s+k].astype(np.float32)      # (k, z_dim)
                tgt = z_arr[s+k].astype(np.float32)       # (z_dim,)
                self.inputs.append(inp)
                self.targets.append(tgt)
                self.conds.append(cond_vec)

        # convert to numpy arrays
        self.inputs = np.stack(self.inputs, axis=0) if len(self.inputs) > 0 else np.zeros((0, k, Z_DIM), dtype=np.float32)
        self.targets = np.stack(self.targets, axis=0) if len(self.targets) > 0 else np.zeros((0, Z_DIM), dtype=np.float32)
        self.conds = np.stack(self.conds, axis=0) if len(self.conds) > 0 else np.zeros((0, COND_DIM), dtype=np.float32)
        print(f"DP dataset built. samples: {self.inputs.shape[0]}, each input shape: {self.inputs.shape[1:]}")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # return tensors: (k, z_dim), (cond_dim,), (z_dim,)
        return (torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.conds[idx]), torch.from_numpy(self.targets[idx]))

# ------------------- Main training routine -------------------
def train_dp():
    # device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        try:
            print(f"GPU Name: {torch.cuda.get_device_name(0)}, Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        except Exception:
            pass

    # load encoder
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoder state dict not found at {ENCODER_PATH}. Please provide trained encoder.pt")

    # instantiate encoder architecture consistent with AE training:
    encoder = Encoder(input_dim=FEATURE_DIM, hidden_size=128, num_layers=2, z_dim=Z_DIM).to(device)
    encoder_state = torch.load(ENCODER_PATH, map_location=device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    # Precompute z for all windows
    all_z_per_file, all_emotion_idx_per_file, all_level_per_file = build_z_sequences(encoder, device)

    # build DP dataset
    dp_dataset = DPZDataset(all_z_per_file, all_emotion_idx_per_file, all_level_per_file, k=K)
    if len(dp_dataset) == 0:
        raise RuntimeError("No DP samples created. Check your data and SEQ_LEN/K settings.")

    # split 90%/10%
    total = len(dp_dataset)
    train_n = int(0.9 * total)
    val_n = total - train_n
    train_ds, val_ds = random_split(dp_dataset, [train_n, val_n])
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # instantiate DP model
    dp_model = TemporalTransformer(z_dim=Z_DIM, cond_dim=COND_DIM, model_dim=MODEL_DIM,
                                   nhead=NHEAD, num_layers=NUM_TRANSFORMER_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(dp_model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # training loop
    start_time = time.time()
    best_val = float('inf')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        for epoch in range(1, EPOCHS + 1):
            dp_model.train()
            train_loss = 0.0
            n_train = 0
            for x_z, cond, y_z in train_loader:
                # shapes:
                #   x_z: (batch, k, Z_DIM)
                #   cond: (batch, COND_DIM)
                #   y_z: (batch, Z_DIM)
                x_z = x_z.to(device)
                cond = cond.to(device)
                y_z = y_z.to(device)

                # expand cond to every timestep and concat
                cond_exp = cond.unsqueeze(1).repeat(1, K, 1)  # (batch, k, COND_DIM)
                inp = torch.cat([x_z, cond_exp], dim=-1)      # (batch, k, Z_DIM+COND_DIM)

                optimizer.zero_grad()
                y_pred = dp_model(inp)                        # (batch, Z_DIM)
                loss = criterion(y_pred, y_z)
                loss.backward()
                optimizer.step()

                b = x_z.size(0)
                train_loss += loss.item() * b
                n_train += b

            train_loss /= max(1, n_train)

            # validation
            dp_model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for x_z, cond, y_z in val_loader:
                    x_z = x_z.to(device)
                    cond = cond.to(device)
                    y_z = y_z.to(device)
                    cond_exp = cond.unsqueeze(1).repeat(1, K, 1)
                    inp = torch.cat([x_z, cond_exp], dim=-1)
                    y_pred = dp_model(inp)
                    loss = criterion(y_pred, y_z)
                    b = x_z.size(0)
                    val_loss += loss.item() * b
                    n_val += b
            val_loss /= max(1, n_val)

            print(f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

            # track best
            if val_loss < best_val:
                best_val = val_loss
                # save best intermediate model (timestamped)
                best_path = MODEL_DIR / f"dp_model_best_{timestamp}.pt"
                torch.save(dp_model.state_dict(), best_path)

    except KeyboardInterrupt:
        print("Training interrupted by user (KeyboardInterrupt). Will save current model state...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training finished! Total time: {total_time:.2f} seconds")

    # final save (timestamped)
    final_path = MODEL_DIR / f"dp_model_final_{timestamp}.pt"
    torch.save(dp_model.state_dict(), final_path)
    print(f"Saved final DP model to: {final_path}")

    # save parameter file
    param_path = MODEL_DIR / f"parameter_DP.txt"
    with open(param_path, "w") as f:
        f.write(f"BASE_DIR: {BASE_DIR}\n")
        f.write(f"DATA_DIR: {DATA_DIR}\n")
        f.write(f"SEQ_LEN: {SEQ_LEN}\n")
        f.write(f"FEATURE_DIM: {FEATURE_DIM}\n")
        f.write(f"Z_DIM: {Z_DIM}\n")
        f.write(f"K: {K}\n")
        f.write(f"COND_DIM: {COND_DIM} (7-d one-hot emotion + 1-d level [0,1])\n")
        f.write(f"MODEL_DIM: {MODEL_DIM}\n")
        f.write(f"NHEAD: {NHEAD}\n")
        f.write(f"NUM_TRANSFORMER_LAYERS: {NUM_TRANSFORMER_LAYERS}\n")
        f.write(f"FF_DIM: {FF_DIM}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
        f.write(f"ENCODER_PATH: {ENCODER_PATH}\n")
        f.write(f"Best Val Loss: {best_val}\n")
        f.write(f"Total training time (s): {total_time:.2f}\n")

    print(f"Saved training parameters to: {param_path}")
    return final_path

# ------------------- Entry -------------------
if __name__ == "__main__":
    # recommended on Windows when using multiple workers
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    final_model = train_dp()
