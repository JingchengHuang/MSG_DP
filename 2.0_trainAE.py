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

SEQ_LEN = 1  # 每次输入一帧数据
FEATURE_DIM = 32
Z_DIM = 16  # 潜向量维度设置为16
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
HIDDEN_SIZE = 128   # LSTM隐藏单元数
NUM_LAYERS = 2      # Encoder和Decoder的LSTM层数

# ------------------- 数据集 -------------------
class RobotDataset(Dataset):
    def __init__(self, data_dir, seq_len=1):  # 修改为 seq_len=1
        self.seq_len = seq_len
        self.data_list = []

        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                values = df.values[:, :FEATURE_DIM]  # 取32列
                num_frames = values.shape[0]
                if num_frames < seq_len:
                    continue
                # 随机切片，每次取一帧
                for start_idx in range(num_frames - seq_len):
                    input_seq = values[start_idx:start_idx + seq_len]
                    target = values[start_idx + seq_len]  # 下一帧作为目标
                    self.data_list.append((input_seq.astype(np.float32), target.astype(np.float32)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        return torch.tensor(x), torch.tensor(y)

# ------------------- 模型 -------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=32, z_dim=16):  # z_dim改为16
        super().__init__()
        self.fc = nn.Linear(input_dim, z_dim)  # 通过全连接层直接映射到潜向量

    def forward(self, x):
        z = self.fc(x)  # 每一帧直接通过全连接层得到潜向量
        return z

class Decoder(nn.Module):
    def __init__(self, z_dim=16, output_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, output_dim)  # 潜向量直接映射到控制参数

    def forward(self, z):
        out = self.fc(z)  # 将潜向量z映射为控制参数
        # 激活函数，确保控制参数在合理范围内
        out[:, :29] = torch.sigmoid(out[:, :29])  # 控制参数的范围 [0, 1]
        out[:, 29:] = torch.tanh(out[:, 29:])  # 控制参数的范围 [-0.8, 0.6] 或 [-0.3, 0.3]
        return out.squeeze(1)

if __name__ == "__main__":
    # ------------------- 设备 -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}, Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

    # ------------------- 数据加载 -------------------
    dataset = RobotDataset(DATA_DIR, seq_len=SEQ_LEN)  # 使用seq_len=1
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ------------------- 初始化模型 -------------------
    encoder = Encoder(input_dim=FEATURE_DIM, z_dim=Z_DIM).to(device)
    decoder = Decoder(z_dim=Z_DIM, output_dim=FEATURE_DIM).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    criterion = nn.MSELoss()

    # ------------------- 训练循环 -------------------
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        # 训练
        encoder.train()
        decoder.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            z = encoder(x_batch)
            y_pred = decoder(z)
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
                y_pred = decoder(z)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

    end_time = time.time()
    print(f"Training finished! Total time: {end_time - start_time:.2f} seconds")

    # ------------------- 保存模型 -------------------
    # 只在训练成功完成时才创建MODEL_DIR文件夹
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
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
