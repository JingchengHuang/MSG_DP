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
MODEL_DIR = BASE_DIR / "model" / "20251110_161351_model"

# data/encoding params (must match AE training)
SEQ_LEN = 1          # frames per window for encoder -> one z (changed to 1 to match AE)
FEATURE_DIM = 32     # raw control dim
Z_DIM = 16           # latent dim from encoder (changed to match AE)
K = 20                # number of historical z tokens used as input to DP
COND_DIM = 8         # 7-d one-hot emotion + 1-d level

# DP model / training hyperparams
MODEL_DIM = 128
NHEAD = 8
NUM_TRANSFORMER_LAYERS = 3
FF_DIM = 256
DROPOUT = 0.1

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50

NUM_WORKERS = 4      # DataLoader workers
PIN_MEMORY = True

ENCODER_PATH = MODEL_DIR / "encoder.pt"  # Make sure this points to your trained encoder

# ------------------- Encoder class (must match saved encoder architecture) -------------------
# Modified to match the new AE encoder structure (frame-by-frame)
class Encoder(nn.Module):
    def __init__(self, input_dim=32, z_dim=16):  # z_dim改为16 to match AE
        super().__init__()
        self.fc = nn.Linear(input_dim, z_dim)  # 通过全连接层直接映射到潜向量

    def forward(self, x):
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        if x.dim() == 2:
            # If input is (batch, input_dim), add a sequence dimension
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        # For frame-by-frame encoding, we process each frame independently
        batch_size, seq_len, input_dim = x.shape
        x = x.view(-1, input_dim)  # (batch*seq_len, input_dim)
        z = self.fc(x)  # (batch*seq_len, z_dim)
        z = z.view(batch_size, seq_len, -1)  # (batch, seq_len, z_dim)
        # Return only the last frame's z if seq_len > 1, or the z if seq_len = 1
        return z[:, -1, :] if seq_len > 1 else z.squeeze(1)  # (batch, z_dim)

# ------------------- DP LSTM Model -------------------
class DPLSTM(nn.Module):
    def __init__(self, z_dim=Z_DIM, cond_dim=COND_DIM, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = z_dim + cond_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, z_dim)
        
    def forward(self, z_seq_cond):
        # z_seq_cond: (batch, k, z_dim + cond_dim)
        lstm_out, _ = self.lstm(z_seq_cond)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        z_pred = self.fc(last_output)     # (batch, z_dim)
        return z_pred

# ------------------- Preprocessing: make z sequences using encoder -------------------
def build_z_sequences(encoder, device):
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
                df = pd.read_csv(fpath, sep=None, engine='python')

            values = df.values[:, :FEATURE_DIM]
            num_frames = values.shape[0]
            num_windows = num_frames - SEQ_LEN

            if num_windows <= 0:
                print(f"\n[ERROR] File '{fname}' has too few frames ({num_frames}) for SEQ_LEN={SEQ_LEN}.")
                raise ValueError(f"Invalid frame count in file: {fname}")

            # For the new frame-by-frame encoder, we need to process each frame individually
            # but we still need to create sequences for DP training
            if SEQ_LEN == 1:
                # Process each frame individually
                z_list = []
                BATCH_ENC = 256
                for i0 in range(0, num_frames, BATCH_ENC):
                    batch_frames = values[i0:i0 + BATCH_ENC].astype(np.float32)
                    batch_tensor = torch.from_numpy(batch_frames).to(device)
                    # Add sequence dimension for compatibility with encoder
                    batch_tensor = batch_tensor.unsqueeze(1)  # (batch, 1, FEATURE_DIM)
                    z_batch = encoder(batch_tensor)
                    z_list.append(z_batch.cpu().numpy())
                z_arr = np.concatenate(z_list, axis=0)
            else:
                # Use sliding window approach for SEQ_LEN > 1
                try:
                    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=(SEQ_LEN, FEATURE_DIM))[0]
                except ValueError as e:
                    print(f"\n[ERROR] sliding_window_view() failed for file: {fname}")
                    raise e  # re-raise to terminate program immediately

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
            total_windows += z_arr.shape[0]  # Number of z vectors

    print(f"Preprocessing done. Files processed: {len(all_z_per_file)}, total windows (z vectors): {total_windows}")
    return all_z_per_file, all_emotion_idx_per_file, all_level_per_file

# ------------------- Utilities: parse filename for emotion & level -------------------
# Reuse the same function from 2.1_trainDP_ori.py
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

# ------------------- DP dataset -------------------
class DPZDataset(Dataset):
    def __init__(self, all_z_per_file, all_emotion_idx_per_file, all_level_per_file, k=K):
        self.k = k
        self.z_dim = Z_DIM
        self.cond_dim = COND_DIM
        self.inputs = []   # will store np arrays
        self.targets = []
        self.conds = []

        for z_arr, emo_idx, lvl in zip(all_z_per_file, all_emotion_idx_per_file, all_level_per_file):
            n = z_arr.shape[0]
            max_s = n - (k + 1) + 1  # n - (k+1) + 1 == n - k
            if max_s <= 0:
                continue
            onehot = np.zeros(7, dtype=np.float32)
            onehot[emo_idx] = 1.0
            level_arr = np.array([lvl], dtype=np.float32)
            cond_vec = np.concatenate([onehot, level_arr], axis=0)

            for s in range(0, n - (k + 1) + 1):
                inp = z_arr[s:s+k].astype(np.float32)      # (k, z_dim)
                tgt = z_arr[s+k].astype(np.float32)       # (z_dim,)
                self.inputs.append(inp)
                self.targets.append(tgt)
                self.conds.append(cond_vec)

        self.inputs = np.stack(self.inputs, axis=0)
        self.targets = np.stack(self.targets, axis=0)
        self.conds = np.stack(self.conds, axis=0)
        print(f"DP dataset built. samples: {self.inputs.shape[0]}")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.conds[idx]), torch.from_numpy(self.targets[idx]))

# ------------------- Main training routine -------------------
def train_dp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoder state dict not found at {ENCODER_PATH}. Please provide trained encoder.pt")

    encoder = Encoder(input_dim=FEATURE_DIM, z_dim=Z_DIM).to(device)
    encoder_state = torch.load(ENCODER_PATH, map_location=device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    all_z_per_file, all_emotion_idx_per_file, all_level_per_file = build_z_sequences(encoder, device)
    dp_dataset = DPZDataset(all_z_per_file, all_emotion_idx_per_file, all_level_per_file, k=K)

    if len(dp_dataset) == 0:
        raise RuntimeError("No DP samples created. Check your data and SEQ_LEN/K settings.")

    total = len(dp_dataset)
    train_n = int(0.9 * total)
    val_n = total - train_n
    train_ds, val_ds = random_split(dp_dataset, [train_n, val_n])
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    dp_model = DPLSTM(z_dim=Z_DIM, cond_dim=COND_DIM, hidden_size=MODEL_DIM,
                      num_layers=NUM_TRANSFORMER_LAYERS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(dp_model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    start_time = time.time()
    best_val = float('inf')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        for epoch in range(1, EPOCHS + 1):
            dp_model.train()
            train_loss = 0.0
            n_train = 0
            for x_z, cond, y_z in train_loader:
                x_z = x_z.to(device)
                cond = cond.to(device)
                y_z = y_z.to(device)

                cond_exp = cond.unsqueeze(1).repeat(1, K, 1)
                inp = torch.cat([x_z, cond_exp], dim=-1)

                optimizer.zero_grad()
                y_pred = dp_model(inp)
                loss = criterion(y_pred, y_z)
                loss.backward()
                optimizer.step()

                b = x_z.size(0)
                train_loss += loss.item() * b
                n_train += b

            train_loss /= max(1, n_train)

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

            if val_loss < best_val:
                best_val = val_loss
                best_path = MODEL_DIR / f"dp_model_best_{timestamp}.pt"
                torch.save(dp_model.state_dict(), best_path)

    except KeyboardInterrupt:
        print("Training interrupted by user (KeyboardInterrupt). Will save current model state...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training finished! Total time: {total_time:.2f} seconds")

    final_path = MODEL_DIR / f"dp_model_final_{timestamp}.pt"
    torch.save(dp_model.state_dict(), final_path)
    print(f"Saved final DP model to: {final_path}")

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
        f.write(f"NUM_LAYERS: {NUM_TRANSFORMER_LAYERS}\n")
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
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    final_model = train_dp()