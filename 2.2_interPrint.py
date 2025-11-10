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
MODEL_DIR = BASE_DIR / "model" / "20251110_161351_model"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 可配置参数 ==========
EMOTION = "angry"    # 情绪类型: 'happy','angry','sad','surprise','disgust','fear','neutral'
LEVEL = 1.0            # 情绪强度: 0.0-1.0

SEQ_LEN = 1            # 新框架中SEQ_LEN为1
HISTORY_K = 10         # 与训练时的K值保持一致

NOISE_LEVEL = 0.01     # 噪声扰动水平
TEMPERATURE = 0.1      # 温度调节参数
GENERATE_FRAMES = 500
FEATURE_DIM = 32
Z_DIM = 16             # 新框架中Z_DIM为16
COND_DIM = 8           # 7-d one-hot + 1-d intensity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {gpu.name}, Total Memory: {gpu.total_memory / 1024 ** 2:.2f} MB")

# ========== Encoder （与新训练框架一致） ==========
class Encoder(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, z_dim=Z_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, z_dim)  # 全连接层直接映射到潜向量

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

# ========== Decoder （与新训练框架一致） ==========
class Decoder(nn.Module):
    def __init__(self, z_dim=Z_DIM, output_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, output_dim)  # 潜向量直接映射到控制参数

    def forward(self, z, temperature=0.0):
        out = self.fc(z)  # 将潜向量z映射为控制参数
        # 添加温度调节采样
        if temperature > 0:
            out = out + torch.randn_like(out) * temperature
        # 激活函数，确保控制参数在合理范围内
        out[:, :29] = torch.sigmoid(out[:, :29])  # 控制参数的范围 [0, 1]
        out[:, 29:] = torch.tanh(out[:, 29:])     # 控制参数的范围
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) - 适配batch_first
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)  # 适配batch_first
        x = x + self.pe[:, :seq_len]
        return x

# ========== DP model matching new training-time architecture ==========
class DPTemporalTransformer(nn.Module):
    def __init__(self, z_dim=Z_DIM, cond_dim=COND_DIM, model_dim=128, nhead=8, num_layers=3, ff_dim=256, dropout=0.1):
        super().__init__()
        self.input_dim = z_dim + cond_dim
        self.model_dim = model_dim
        self.input_fc = nn.Linear(self.input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim, max_len=HISTORY_K + 5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True  # 使用batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(model_dim, z_dim)

    def forward(self, z_seq_cond):
        x = self.input_fc(z_seq_cond)   # (batch, k, model_dim)
        # x = x.permute(1, 0, 2)       # 不再需要，因为使用batch_first
        x = self.pos_enc(x)             # add positional encoding
        x = self.transformer(x)         # (batch, k, model_dim) - 使用batch_first
        last = x[:, -1, :]              # (batch, model_dim) -- last time token representation
        z_pred = self.out_fc(last)      # (batch, z_dim)
        return z_pred

# ========== 加载模型权重 ==========
encoder = Encoder().to(device)
decoder = Decoder().to(device)
dp_model = DPTemporalTransformer().to(device)

encoder.load_state_dict(torch.load(MODEL_DIR / "encoder.pt", map_location=device))
decoder.load_state_dict(torch.load(MODEL_DIR / "decoder.pt", map_location=device))
# load dp model - ensure filename matches your saved file
dp_model.load_state_dict(torch.load(MODEL_DIR / "dp_model_final_20251110_172227.pt", map_location=device))

encoder.eval()
decoder.eval()
dp_model.eval()
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

# ========== 根据指定的EMOTION和LEVEL选择文件并随机取初始帧 ==========
def get_initial_data(emotion, level):
    # 将LEVEL转换为文件名中的'h'或'l'
    level_str = 'h' if level >= 0.5 else 'l'
    
    # 构造文件名模式
    pattern = f"{emotion}_merged_{level_str}_*.csv"
    files = list(DATA_DIR.glob(pattern))
    
    # 如果没有找到精确匹配的文件，尝试寻找其他可能的文件
    if not files:
        # 尝试不区分大小写
        all_files = list(DATA_DIR.glob("*.csv"))
        for f in all_files:
            fname = f.name.lower()
            if emotion.lower() in fname and level_str in fname:
                files.append(f)
    
    if not files:
        # 如果仍然没有找到，尝试寻找emotion相关的文件
        all_files = list(DATA_DIR.glob("*.csv"))
        for f in all_files:
            fname = f.name.lower()
            if emotion.lower() in fname:
                files.append(f)
        
        if not files:
            raise FileNotFoundError(f"No files found for emotion '{emotion}' and level '{level}'")
    
    # 随机选择一个文件
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
    init_seq = get_initial_data(EMOTION, LEVEL)          # (20, 32)
    all_outputs = [init_seq.copy()]

    # encode sliding windows into z (修改为适应新框架)
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
    input_seq = torch.tensor(init_seq[-1:], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 32) - 只取最后一帧

    gen_controls = []
    for step in range(GENERATE_FRAMES):
        # prepare z_hist with cond: expand cond to each timestep and concat
        cond_exp = cond.unsqueeze(1).repeat(1, HISTORY_K, 1)  # (1, HISTORY_K, 8)
        z_hist_cond = torch.cat([z_hist, cond_exp], dim=-1)  # (1, HISTORY_K, z_dim+cond_dim)

        # predict next z with noise perturbation
        z_pred = dp_model(z_hist_cond)  # (1, z_dim)
        # 添加噪声扰动
        z_pred = z_pred + torch.randn_like(z_pred) * NOISE_LEVEL

        # decode (use current input_seq as decoder input)
        ctrl_pred_t = decoder(z_pred, TEMPERATURE)  # (1, 32) - 新框架中decoder只需要z，添加温度调节
        ctrl_pred = ctrl_pred_t.squeeze(0).cpu().numpy()

        # ======== 严格范围限制 ========
        ctrl_pred[:29] = np.clip(ctrl_pred[:29], 0.0, 1.0)
        ctrl_pred[29] = np.clip(ctrl_pred[29], -0.8, 0.6)
        ctrl_pred[30] = np.clip(ctrl_pred[30], -0.3, 0.3)
        ctrl_pred[31] = np.clip(ctrl_pred[31], -0.55, 0.55)

        gen_controls.append(ctrl_pred.astype(np.float32))

        # construct new input_seq for re-encoding:
        # 在新框架中，input_seq只需要最新的控制帧
        input_seq = torch.tensor(ctrl_pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 32)

        # re-encode this frame to get z_recoded and update z_hist
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