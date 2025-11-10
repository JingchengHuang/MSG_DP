# 训练AE，使用10帧训练得到1帧新的
# 不加情感标签，只有上述时间信息
# 得到的encoder和decoder还有潜向量z，用作DP训练


import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------- 配置 -------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "1030data_sy"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = BASE_DIR / "model" / f"{timestamp}_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 10
FEATURE_DIM = 32
Z_DIM = 64
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
HIDDEN_SIZE = 128   # LSTM隐藏单元数
NUM_LAYERS = 2      # Encoder和Decoder的LSTM层数

# ------------------- 数据集 -------------------
class RobotDataset(Dataset):
    def __init__(self, data_dir, seq_len=10):
        self.seq_len = seq_len
        self.data_list = []

        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                values = df.values[:, :FEATURE_DIM]  # 忽略表头，取32列
                num_frames = values.shape[0]
                if num_frames <= seq_len:
                    continue
                # 随机切片
                for start_idx in range(num_frames - seq_len):
                    input_seq = values[start_idx:start_idx + seq_len]
                    target = values[start_idx + seq_len]
                    self.data_list.append((input_seq.astype(np.float32), target.astype(np.float32)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        return torch.tensor(x), torch.tensor(y)

# ------------------- 模型 -------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=32, hidden_size=128, num_layers=2, z_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, z_dim)  # Bi-LSTM

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers*2, batch, hidden)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # 拼接双向最后一层
        z = self.fc(h_last)
        return z

class Decoder(nn.Module):
    def __init__(self, z_dim=64, hidden_size=128, num_layers=2, output_dim=32, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(output_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.z2h = nn.Linear(z_dim, hidden_size)

    def forward(self, z, input_seq):
        # z -> 初始化隐藏状态
        h_0 = self.z2h(z).unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
        c_0 = torch.zeros_like(h_0)
        # 使用输入序列的最后一帧作为Decoder输入
        out, _ = self.lstm(input_seq, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        # 激活
        out[:, :29] = torch.sigmoid(out[:, :29])
        out[:, 29:] = torch.tanh(out[:, 29:])
        return out

if __name__ == "__main__":
    # ------------------- 设备 -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}, Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

    # ------------------- 数据加载 -------------------
    dataset = RobotDataset(DATA_DIR, seq_len=SEQ_LEN)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ------------------- 初始化模型 -------------------
    encoder = Encoder(input_dim=FEATURE_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, z_dim=Z_DIM).to(device)
    decoder = Decoder(z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_dim=FEATURE_DIM, seq_len=SEQ_LEN).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    criterion = nn.MSELoss()

    # ------------------- 训练循环 -------------------
    start_time = time.time()
    for epoch in range(1, EPOCHS+1):
        # 训练
        encoder.train()
        decoder.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            z = encoder(x_batch)
            y_pred = decoder(z, x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证
        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                z = encoder(x_batch)
                y_pred = decoder(z, x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

    end_time = time.time()
    print(f"Training finished! Total time: {end_time - start_time:.2f} seconds")

    # ------------------- 保存模型 -------------------
    torch.save(encoder.state_dict(), MODEL_DIR / "encoder.pt")
    torch.save(decoder.state_dict(), MODEL_DIR / "decoder.pt")

    # ------------------- 保存超参数 -------------------
    param_path = MODEL_DIR / "parameter_AE.txt"
    with open(param_path, "w") as f:
        f.write(f"SEQ_LEN: {SEQ_LEN}\n")
        f.write(f"FEATURE_DIM: {FEATURE_DIM}\n")
        f.write(f"Z_DIM: {Z_DIM}\n")
        f.write(f"HIDDEN_SIZE: {HIDDEN_SIZE}\n")
        f.write(f"NUM_LAYERS: {NUM_LAYERS}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
