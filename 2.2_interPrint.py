import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from pathlib import Path
import time
from datetime import datetime
import os
import re

# ========== 配置 ==========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "1030data_sy"
MODEL_DIR = BASE_DIR / "model" / "1103model"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 可配置参数 ==========
EMOTION = "angry"    # 情绪类型: 'happy','angry','sad','surprise','disgust','fear','neutral'
LEVEL = 0.8            # 情绪强度: 0.0-1.0

SEQ_LEN = 10
HISTORY_K = 3
GENERATE_FRAMES = 500
FEATURE_DIM = 32
Z_DIM = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
COND_DIM = 8   # 7-d one-hot + 1-d intensity (training used cond)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {gpu.name}, Total Memory: {gpu.total_memory / 1024 ** 2:.2f} MB")

# ========== Encoder （与训练一致） ==========
class Encoder(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, z_dim=Z_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, z_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        z = self.fc(h_last)
        return z  # (batch, z_dim)

# ========== Decoder （与你训练时一致） ==========
class Decoder(nn.Module):
    def __init__(self, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_dim=FEATURE_DIM, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(output_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.z2h = nn.Linear(z_dim, hidden_size)

    def forward(self, z, input_seq):
        # z : (batch, z_dim)
        # input_seq: (batch, seq_len, output_dim)
        h_0 = self.z2h(z).unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
        c_0 = torch.zeros_like(h_0).to(z.device)
        out, _ = self.lstm(input_seq, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out[:, :29] = torch.sigmoid(out[:, :29])
        out[:, 29:] = torch.tanh(out[:, 29:])
        return out

# ========== PositionalEncoding (matches training's pos_enc.pe) ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len,1,d_model)
        # save as buffer named 'pe' to match checkpoint key 'pos_enc.pe'
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model) or (batch, seq_len, d_model) depending usage
        # We'll use it consistent with encoder_layer batch_first=False in training
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x

# ========== DP model matching training-time architecture ==========
class DPTemporalTransformer(nn.Module):
    def __init__(self, z_dim=Z_DIM, cond_dim=COND_DIM, model_dim=128, nhead=8, num_layers=3, ff_dim=256, dropout=0.1):
        """
        This matches your checkpoint architecture:
        - num_layers=3
        - pos_enc.pe with shape [8, 1, 128]
        """
        super().__init__()
        self.input_dim = z_dim + cond_dim
        self.model_dim = model_dim
        self.input_fc = nn.Linear(self.input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim, max_len=8)  # match checkpoint exactly
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(model_dim, z_dim)

    def forward(self, z_seq_cond):
        x = self.input_fc(z_seq_cond)   # (batch, k, model_dim)
        x = x.permute(1, 0, 2)          # (k, batch, model_dim)
        x = x + self.pos_enc.pe[:x.size(0)]  # add positional encoding
        x = self.transformer(x)         # (k, batch, model_dim)
        last = x[-1]
        z_pred = self.out_fc(last)
        return z_pred


# ========== 加载模型权重 ==========
encoder = Encoder().to(device)
decoder = Decoder().to(device)
# DP model uses larger internal model_dim and expects cond - match checkpoint
dp_model = DPTemporalTransformer().to(device)

encoder.load_state_dict(torch.load(MODEL_DIR / "encoder.pt", map_location=device))
decoder.load_state_dict(torch.load(MODEL_DIR / "decoder.pt", map_location=device))
# load dp model - ensure filename matches your saved file
dp_model.load_state_dict(torch.load(MODEL_DIR / "dp_model_final_20251103_221528.pt", map_location=device))

encoder.eval(); decoder.eval(); dp_model.eval()
print("Models loaded successfully.")

# ========== helper: build condition vector (7-d one-hot + 1 intensity) ==========
def build_cond_vec(emotion:str, level:float):
    # emotions: ['happy','angry','sad','surprise','disgust','fear','neutral']
    emotions = ['happy','angry','sad','surprise','disgust','fear','neutral']
    onehot = np.zeros(len(emotions), dtype=np.float32)
    try:
        idx = emotions.index(emotion)
    except ValueError:
        idx = 6  # neutral fallback
    onehot[idx] = 1.0
    lvl = float(level)
    return np.concatenate([onehot, np.array([lvl], dtype=np.float32)], axis=0)  # shape (8,)

# ========== 从 neutral_h 文件中随机取初始帧 ==========
def get_neutral_h_data():
    files = [f for f in DATA_DIR.glob("neutral_merged_h_*.csv")]
    if not files:
        raise FileNotFoundError("No neutral_h files found in dataset.")
    fpath = random.choice(files)
    df = pd.read_csv(fpath)
    df = df.dropna()
    values = df.values[:, :FEATURE_DIM].astype(np.float32)
    start = random.randint(0, len(values) - 20)
    init_seq = values[start:start + 20]
    return init_seq

# ========== 主推理过程（使用 cond 拼接到 z） ==========
@torch.no_grad()
def inference_loop():
    init_seq = get_neutral_h_data()                      # (20, 32)
    all_outputs = [init_seq.copy()]

    # encode sliding windows into z
    z_list = []
    for i in range(len(init_seq) - SEQ_LEN + 1):
        x_window = torch.tensor(init_seq[i:i + SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(device)  # (1, SEQ_LEN, 32)
        z = encoder(x_window)  # (1, z_dim)
        z_list.append(z.squeeze(0))
    z_seq = torch.stack(z_list)  # (num_windows, z_dim)

    # initial z_hist
    z_hist = z_seq[-HISTORY_K:].clone().unsqueeze(0).to(device)  # (1, HISTORY_K, z_dim)

    # build condition vector using configurable EMOTION and LEVEL
    cond_np = build_cond_vec(EMOTION, LEVEL)  # (8,)
    cond = torch.from_numpy(cond_np).unsqueeze(0).to(device)  # (1, 8)

    # prepare ctrl buffer for re-encoding (last SEQ_LEN frames)
    input_seq = torch.tensor(init_seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(device)  # (1, SEQ_LEN, 32)

    gen_controls = []
    for step in range(GENERATE_FRAMES):
        # prepare z_hist with cond: expand cond to each timestep and concat
        cond_exp = cond.unsqueeze(1).repeat(1, HISTORY_K, 1)  # (1, HISTORY_K, 8)
        z_hist_cond = torch.cat([z_hist, cond_exp], dim=-1)  # (1, HISTORY_K, z_dim+cond_dim)

        # predict next z
        z_pred = dp_model(z_hist_cond)  # (1, z_dim)

        # decode (use current input_seq as decoder input)
        ctrl_pred_t = decoder(z_pred, input_seq)  # (1, 32)
        ctrl_pred = ctrl_pred_t.squeeze(0).cpu().numpy()

        # ======== 严格范围限制 ========
        ctrl_pred[:29] = np.clip(ctrl_pred[:29], 0.0, 1.0)
        ctrl_pred[29] = np.clip(ctrl_pred[29], -0.8, 0.6)
        ctrl_pred[30] = np.clip(ctrl_pred[30], -0.3, 0.3)
        ctrl_pred[31] = np.clip(ctrl_pred[31], -0.55, 0.55)

        gen_controls.append(ctrl_pred.astype(np.float32))

        # construct new input_seq window for re-encoding:
        # take last SEQ_LEN-1 frames from input_seq (drop oldest) + ctrl_pred
        input_seq_np = input_seq.squeeze(0).cpu().numpy()  # (SEQ_LEN, 32)
        new_window = np.vstack([input_seq_np[1:], ctrl_pred])  # (SEQ_LEN, 32)
        input_seq = torch.tensor(new_window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, SEQ_LEN, 32)

        # re-encode this window to get z_recoded and update z_hist
        z_recoded = encoder(input_seq)  # (1, z_dim)
        z_hist = torch.cat([z_hist[:, 1:, :], z_recoded.unsqueeze(1)], dim=1)  # (1, HISTORY_K, z_dim)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1}/{GENERATE_FRAMES}")

    gen_controls = np.array(gen_controls)
    final_output = np.concatenate([init_seq, gen_controls], axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}_{EMOTION}_{LEVEL}.csv"
    pd.DataFrame(final_output).to_csv(out_path, index=False, header=False)
    print(f"\n✅ Generation complete. Saved to: {out_path}")
    print(f"Output shape: {final_output.shape}")

if __name__ == "__main__":
    start = time.time()
    inference_loop()
    print(f"\nTotal time: {time.time() - start:.2f}s")
