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
MODEL_DIR = BASE_DIR / "model" / "20251112_183734_model"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 可配置参数 ==========
EMOTION = "angry"    # 情绪类型: 'happy','angry','sad','surprise','disgust','fear','neutral'
LEVEL = 1.0         # 情绪强度: 0.0-1.0

HISTORY_K = 11       # 使用11帧历史数据来计算10帧差值
PREDICT_STEPS = 1    # 每次预测1帧差值
GENERATE_FRAMES = 200  # 生成批次数量（每个批次1帧差值，总共1000帧控制参数）
FEATURE_DIM = 32
COND_DIM = 8         # 7-d one-hot + 1-d intensity

NOISE_LEVEL = 0.01   # 噪声扰动水平
TEMPERATURE = 0.01    # 温度调节参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {gpu.name}, Total Memory: {gpu.total_memory / 1024 ** 2:.2f} MB")

# ========== DP LSTM model (differences prediction) ==========
class DPLSTM(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, cond_dim=COND_DIM, hidden_size=128, num_layers=3, dropout=0.1, predict_steps=1):
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
        # 预测一个时间步
        output = self.fc(last_hidden)  # (batch, input_dim * predict_steps)
        # 重塑为(batch, predict_steps, input_dim)
        ctrl_pred = output.view(-1, self.predict_steps, self.input_dim)  # (batch, predict_steps, input_dim)
        return ctrl_pred

# ========== 加载模型权重 ==========
dp_model = DPLSTM(predict_steps=PREDICT_STEPS).to(device)
# load dp model - ensure filename matches your saved file
dp_model.load_state_dict(torch.load(MODEL_DIR / "dp_model_final_20251112_183736.pt", map_location=device))

dp_model.eval()
print("DP model loaded successfully.")

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
    start = random.randint(0, len(values) - HISTORY_K)
    init_seq = values[start:start + HISTORY_K]
    return init_seq

# ========== 主推理过程（使用差值预测机制） ==========
@torch.no_grad()
def inference_loop():
    # 获取初始控制序列
    init_seq = get_initial_data(EMOTION, LEVEL)          # (HISTORY_K, 32)
    
    # 初始化历史控制序列
    ctrl_hist = torch.tensor(init_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, HISTORY_K, 32)
    
    # build condition vector using configurable EMOTION and LEVEL
    cond_np = build_cond_vec(EMOTION, LEVEL)  # (8,)
    cond = torch.from_numpy(cond_np).unsqueeze(0).to(device)  # (1, 8)
    
    # 将初始序列展平为1D数组列表
    gen_controls = []
    for i in range(init_seq.shape[0]):
        gen_controls.append(init_seq[i].copy())
    
    # 获取历史控制序列的最后一帧，用于重建控制参数
    last_ctrl_frame = init_seq[-1].copy()
    
    for step in range(GENERATE_FRAMES):
        # 计算历史帧的差值 (11帧 -> 10个差值)
        ctrl_diffs = torch.diff(ctrl_hist, dim=1)  # (1, HISTORY_K-1, 32)
        
        # prepare ctrl_diffs with cond: expand cond to each timestep and concat
        cond_exp = cond.unsqueeze(1).repeat(1, HISTORY_K-1, 1)  # (1, HISTORY_K-1, 8)
        ctrl_diffs_cond = torch.cat([ctrl_diffs, cond_exp], dim=-1)  # (1, HISTORY_K-1, 32+8)
        
        # predict next control differences
        diff_pred = dp_model(ctrl_diffs_cond)  # (1, PREDICT_STEPS, 32)
        
        # 添加噪声扰动和温度调节采样
        if NOISE_LEVEL > 0 or TEMPERATURE > 0:
            noise = torch.randn_like(diff_pred) * (NOISE_LEVEL + TEMPERATURE)
            diff_pred = diff_pred + noise
            
        # 将预测的差值转换为控制参数
        diff_pred_np = diff_pred.squeeze(0).cpu().numpy()  # (PREDICT_STEPS, 32)
        
        # 重建控制参数: 从last_ctrl_frame开始累加差值
        ctrl_pred_frames = []
        current_frame = last_ctrl_frame.copy()
        
        # 应用差值限制
        diff_frame = diff_pred_np[0].copy()
        diff_frame[:29] = np.clip(diff_frame[:29], -0.1, 0.1)  # 限制差值范围
        diff_frame[29] = np.clip(diff_frame[29], -0.1, 0.1)
        diff_frame[30] = np.clip(diff_frame[30], -0.1, 0.1)
        diff_frame[31] = np.clip(diff_frame[31], -0.1, 0.1)
        
        # 累加差值得到新的控制参数
        current_frame = current_frame + diff_frame
        
        # 应用控制参数范围限制
        current_frame[:29] = np.clip(current_frame[:29], 0.0, 1.0)
        current_frame[29] = np.clip(current_frame[29], -0.8, 0.6)
        current_frame[30] = np.clip(current_frame[30], -0.3, 0.3)
        current_frame[31] = np.clip(current_frame[31], -0.55, 0.55)
        
        ctrl_pred_frames.append(current_frame.copy())
        diff_pred_np[0] = diff_frame  # 保存处理后的差值
        
        ctrl_pred_np = np.array(ctrl_pred_frames)  # (PREDICT_STEPS, 32)
        
        # 直接添加预测的帧到生成列表
        gen_controls.append(ctrl_pred_np[0].copy())
        
        # 更新历史控制序列：移除前1帧，添加新预测的帧
        new_frame = ctrl_pred_np[0:1]  # 取第1帧
        last_ctrl_frame = new_frame[0].copy()
        
        new_frame_tensor = torch.tensor(new_frame, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 32)
        ctrl_hist = torch.cat([ctrl_hist[:, 1:, :], new_frame_tensor], dim=1)  # (1, HISTORY_K, 32)
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Batch {step+1}/{GENERATE_FRAMES} (Generated {len(gen_controls)} frames)")
    
    gen_controls = np.array(gen_controls)
    final_output = gen_controls
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}_{EMOTION}_{LEVEL}.csv"
    pd.DataFrame(final_output).to_csv(out_path, index=False, header=False)
    print(f"✅ Generation complete. Saved to: {out_path}")
    print(f"Output shape: {final_output.shape}")

if __name__ == "__main__":
    start = time.time()
    inference_loop()
    print(f"\nTotal time: {time.time() - start:.2f}s")