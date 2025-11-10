import os
import time
from datetime import datetime
from pathlib import Path
import re

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

# data/encoding params
SEQ_LEN = 32         # raw control dim
K = 20               # number of historical control tokens used as input to DP
COND_DIM = 8         # 7-d one-hot emotion + 1-d level
PREDICT_STEPS = 5    # number of future steps to predict

# DP model / training hyperparams
MODEL_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.1

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100

NUM_WORKERS = 0      # DataLoader workers (set to 0 for Windows compatibility)
PIN_MEMORY = False

# ------------------- DP LSTM Model (direct on control parameters) -------------------
class DPLSTM(nn.Module):
    def __init__(self, input_dim=SEQ_LEN, cond_dim=COND_DIM, hidden_size=MODEL_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, predict_steps=5):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_steps = predict_steps  # 预测步数
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim + self.cond_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, input_dim * predict_steps)
        
    def forward(self, ctrl_seq_cond):
        # ctrl_seq_cond: (batch, k, input_dim + cond_dim)
        lstm_out, (h_n, c_n) = self.lstm(ctrl_seq_cond)
        # 取最后一个时间步的隐藏状态
        last_hidden = h_n[-1]  # (batch, hidden_size)
        # 预测多个时间步
        output = self.fc(last_hidden)  # (batch, input_dim * predict_steps)
        # 重塑为(batch, predict_steps, input_dim)
        ctrl_pred = output.view(-1, self.predict_steps, self.input_dim)  # (batch, predict_steps, input_dim)
        return ctrl_pred

# ------------------- Utilities: parse filename for emotion & level -------------------
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

    # 如果没匹配到情络名，再用正则尝试一次
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

# ------------------- DP dataset (direct on control parameters) -------------------
class DPControlDataset(Dataset):
    def __init__(self, all_ctrl_per_file, all_emotion_idx_per_file, all_level_per_file, k=K, predict_steps=PREDICT_STEPS):
        self.k = k
        self.predict_steps = predict_steps
        self.ctrl_dim = SEQ_LEN
        self.cond_dim = COND_DIM
        self.inputs = []   # will store np arrays
        self.targets = []
        self.conds = []

        for ctrl_arr, emo_idx, lvl in zip(all_ctrl_per_file, all_emotion_idx_per_file, all_level_per_file):
            n = ctrl_arr.shape[0]
            max_s = n - (k + predict_steps) + 1  # n - (k+predict_steps) + 1
            if max_s <= 0:
                continue
            onehot = np.zeros(7, dtype=np.float32)
            onehot[emo_idx] = 1.0
            level_arr = np.array([lvl], dtype=np.float32)
            cond_vec = np.concatenate([onehot, level_arr], axis=0)

            for s in range(0, max_s):
                inp = ctrl_arr[s:s+k].astype(np.float32)      # (k, ctrl_dim)
                tgt = ctrl_arr[s+k:s+k+predict_steps].astype(np.float32)  # (predict_steps, ctrl_dim)
                self.inputs.append(inp)
                self.targets.append(tgt)
                self.conds.append(cond_vec)

        self.inputs = np.stack(self.inputs, axis=0)
        self.targets = np.stack(self.targets, axis=0)
        self.conds = np.stack(self.conds, axis=0)
        print(f"DP control dataset built. samples: {self.inputs.shape[0]}")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.conds[idx]), torch.from_numpy(self.targets[idx]))

# ------------------- Preprocessing: make control sequences -------------------
def build_control_sequences():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    if len(files) == 0:
        raise RuntimeError(f"No csv files found in {DATA_DIR}")

    all_ctrl_per_file = []
    all_emotion_idx_per_file = []
    all_level_per_file = []
    total_windows = 0

    for fname in files:
        fpath = DATA_DIR / fname
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            df = pd.read_csv(fpath, sep=None, engine='python')

        values = df.values[:, :SEQ_LEN]
        num_frames = values.shape[0]
        
        # Skip files with too few frames
        if num_frames <= K:
            print(f"\n[SKIP] File '{fname}' has too few frames ({num_frames}) for K={K}.")
            continue

        emo_idx, lvl = parse_emotion_level_from_name(fname)
        if emo_idx is None or lvl is None:
            continue

        all_ctrl_per_file.append(values.astype(np.float32))
        all_emotion_idx_per_file.append(int(emo_idx))
        all_level_per_file.append(float(lvl))
        total_windows += num_frames

    print(f"Preprocessing done. Files processed: {len(all_ctrl_per_file)}, total frames: {total_windows}")
    return all_ctrl_per_file, all_emotion_idx_per_file, all_level_per_file

# ------------------- Main training routine -------------------
def train_dp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Precompute control sequences
    all_ctrl_per_file, all_emotion_idx_per_file, all_level_per_file = build_control_sequences()
    dp_dataset = DPControlDataset(all_ctrl_per_file, all_emotion_idx_per_file, all_level_per_file, k=K)

    if len(dp_dataset) == 0:
        raise RuntimeError("No DP samples created. Check your data and K settings.")

    total = len(dp_dataset)
    train_n = int(0.9 * total)
    val_n = total - train_n
    train_ds, val_ds = random_split(dp_dataset, [train_n, val_n])
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)

    dp_model = DPLSTM(input_dim=SEQ_LEN, cond_dim=COND_DIM, hidden_size=MODEL_DIM,
                      num_layers=NUM_LAYERS, dropout=DROPOUT, predict_steps=PREDICT_STEPS).to(device)
    optimizer = torch.optim.Adam(dp_model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    start_time = time.time()
    best_val = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        for epoch in range(1, EPOCHS + 1):
            dp_model.train()
            train_loss = 0.0
            n_train = 0
            for x_ctrl, cond, y_ctrl in train_loader:
                x_ctrl = x_ctrl.to(device)
                cond = cond.to(device)
                y_ctrl = y_ctrl.to(device)

                cond_exp = cond.unsqueeze(1).repeat(1, K, 1)
                inp = torch.cat([x_ctrl, cond_exp], dim=-1)

                optimizer.zero_grad()
                y_pred = dp_model(inp)
                loss = criterion(y_pred.view(-1, SEQ_LEN), y_ctrl.view(-1, SEQ_LEN))
                loss.backward()
                optimizer.step()

                b = x_ctrl.size(0)
                train_loss += loss.item() * b
                n_train += b

            train_loss /= max(1, n_train)

            dp_model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for x_ctrl, cond, y_ctrl in val_loader:
                    x_ctrl = x_ctrl.to(device)
                    cond = cond.to(device)
                    y_ctrl = y_ctrl.to(device)
                    cond_exp = cond.unsqueeze(1).repeat(1, K, 1)
                    inp = torch.cat([x_ctrl, cond_exp], dim=-1)
                    y_pred = dp_model(inp)
                    loss = criterion(y_pred.view(-1, SEQ_LEN), y_ctrl.view(-1, SEQ_LEN))
                    b = x_ctrl.size(0)
                    val_loss += loss.item() * b
                    n_val += b
            val_loss /= max(1, n_val)

            print(f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                # 确保MODEL_DIR文件夹存在
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                best_path = MODEL_DIR / f"dp_model_best_{timestamp}.pt"
                torch.save(dp_model.state_dict(), best_path)

    except KeyboardInterrupt:
        print("Training interrupted by user (KeyboardInterrupt). Will save current model state...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training finished! Total time: {total_time:.2f} seconds")

    # 只在训练成功完成时才创建MODEL_DIR文件夹
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    final_path = MODEL_DIR / f"dp_model_final_{timestamp}.pt"
    torch.save(dp_model.state_dict(), final_path)
    print(f"Saved final DP model to: {final_path}")

    param_path = MODEL_DIR / f"parameter_DP_only.txt"
    with open(param_path, "w") as f:
        f.write(f"BASE_DIR: {BASE_DIR}\n")
        f.write(f"DATA_DIR: {DATA_DIR}\n")
        f.write(f"SEQ_LEN: {SEQ_LEN}\n")
        f.write(f"K: {K}\n")
        f.write(f"PREDICT_STEPS: {PREDICT_STEPS}\n")
        f.write(f"COND_DIM: {COND_DIM} (7-d one-hot emotion + 1-d level [0,1])\n")
        f.write(f"MODEL_DIM: {MODEL_DIM}\n")
        f.write(f"NUM_LAYERS: {NUM_LAYERS}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
        f.write(f"Best Val Loss: {best_val}\n")
        f.write(f"Total training time (s): {total_time:.2f}\n")

    print(f"Saved training parameters to: {param_path}")
    return final_path

# ------------------- Entry -------------------
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    final_model = train_dp()