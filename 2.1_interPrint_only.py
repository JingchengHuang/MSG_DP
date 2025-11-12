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
MODEL_DIR = BASE_DIR / "model" / "20251112_172619_model"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 可配置参数 ==========
EMOTION = "angry"    # 情绪类型: 'happy','angry','sad','surprise','disgust','fear','neutral'
LEVEL = 0.2         # 情绪强度: 0.0-1.0

HISTORY_K = 10       # 使用10帧历史数据
PREDICT_STEPS = 5    # 每次预测5帧新数据
GENERATE_FRAMES = 200  # 生成批次数量（每个批次5帧，总共500帧）
FEATURE_DIM = 32
COND_DIM = 8         # 7-d one-hot + 1-d intensity

NOISE_LEVEL = 0.01   # 噪声扰动水平
TEMPERATURE = 0.01    # 温度调节参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {gpu.name}, Total Memory: {gpu.total_memory / 1024 ** 2:.2f} MB")

# ========== DP LSTM model (direct on control parameters) ==========
class DPLSTM(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, cond_dim=COND_DIM, hidden_size=128, num_layers=3, dropout=0.1, predict_steps=5):
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

# ========== 加载模型权重 ==========
dp_model = DPLSTM(predict_steps=PREDICT_STEPS).to(device)
# load dp model - ensure filename matches your saved file
dp_model.load_state_dict(torch.load(MODEL_DIR / "dp_model_final_20251112_172621.pt", map_location=device))

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

# ========== 主推理过程（直接在控制参数上进行DP） ==========
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
    
    # 用于存储历史预测结果，用于重叠帧均值计算
    prediction_history = []
    
    for step in range(GENERATE_FRAMES):
        # prepare ctrl_hist with cond: expand cond to each timestep and concat
        cond_exp = cond.unsqueeze(1).repeat(1, HISTORY_K, 1)  # (1, HISTORY_K, 8)
        ctrl_hist_cond = torch.cat([ctrl_hist, cond_exp], dim=-1)  # (1, HISTORY_K, 32+8)
        
        # predict next control parameters
        ctrl_pred = dp_model(ctrl_hist_cond)  # (1, PREDICT_STEPS, 32)
        
        # 添加噪声扰动和温度调节采样
        if NOISE_LEVEL > 0 or TEMPERATURE > 0:
            noise = torch.randn_like(ctrl_pred) * (NOISE_LEVEL + TEMPERATURE)
            ctrl_pred = ctrl_pred + noise
            
        # 严格范围限制
        ctrl_pred_np = ctrl_pred.squeeze(0).cpu().numpy()  # (PREDICT_STEPS, 32)
        for i in range(PREDICT_STEPS):
            frame = ctrl_pred_np[i].copy()
            frame[:29] = np.clip(frame[:29], 0.0, 1.0)
            frame[29] = np.clip(frame[29], -0.8, 0.6)
            frame[30] = np.clip(frame[30], -0.3, 0.3)
            frame[31] = np.clip(frame[31], -0.55, 0.55)
            ctrl_pred_np[i] = frame.astype(np.float32)
        
        # 根据不同阶段处理帧
        if step < 5:
            # 前五次：逐步增加均值帧数
            prediction_history.append(ctrl_pred_np.copy())
            
            if step == 0:
                # 第1次：直接添加第1帧
                gen_controls.append(ctrl_pred_np[0].copy())
            elif step == 1:
                # 第2次：第1帧与上一次的第2帧均值
                avg_frame = (ctrl_pred_np[0] + prediction_history[-2][1]) / 2.0
                gen_controls.append(avg_frame.copy())
            elif step == 2:
                # 第3次：第1帧与上一次的第2帧、上上次的第3帧均值
                avg_frame = (ctrl_pred_np[0] + prediction_history[-2][1] + prediction_history[-3][2]) / 3.0
                gen_controls.append(avg_frame.copy())
            elif step == 3:
                # 第4次：第1帧与上一次的第2帧、上上次的第3帧、上上上次的第4帧均值
                avg_frame = (ctrl_pred_np[0] + prediction_history[-2][1] + prediction_history[-3][2] + prediction_history[-4][3]) / 4.0
                gen_controls.append(avg_frame.copy())
            elif step == 4:
                # 第5次：第1帧与上一次的第2帧、上上次的第3帧、上上上次的第4帧、上上上上次的第5帧均值
                avg_frame = (ctrl_pred_np[0] + prediction_history[-2][1] + prediction_history[-3][2] + prediction_history[-4][3] + prediction_history[-5][4]) / 5.0
                gen_controls.append(avg_frame.copy())
        else:
            # 从第五次开始：进行完整的均值计算（每1帧做均值）
            # 保存当前预测结果用于后续均值计算
            prediction_history.append(ctrl_pred_np.copy())
            
            # 保持只保留最近5次预测
            if len(prediction_history) > 5:
                prediction_history.pop(0)
            
            # 如果有足够的历史预测结果，进行重叠帧均值计算
            if len(prediction_history) >= 5:
                # 对每1帧进行重叠均值计算
                # 当前预测的第1帧
                current_frame = prediction_history[-1][0]
                # 上一次预测的第2帧
                prev_frame = prediction_history[-2][1]
                # 上上次预测的第3帧
                prev_prev_frame = prediction_history[-3][2]
                # 上上上次预测的第4帧
                prev_prev_prev_frame = prediction_history[-4][3]
                # 上上上上次预测的第5帧
                prev_prev_prev_prev_frame = prediction_history[-5][4]
                
                # 计算均值
                avg_frame = (current_frame + prev_frame + prev_prev_frame + 
                           prev_prev_prev_frame + prev_prev_prev_prev_frame) / 5.0
                gen_controls.append(avg_frame.copy())
        
        # 更新历史控制序列：移除前1帧，添加新预测的第1帧（或均值帧）
        if step < 5:
            # 前五次：使用当前推理的第1帧更新滑窗
            new_frame = ctrl_pred_np[0:1]  # 取第1帧
        else:
            # 从第五次开始：使用均值帧更新滑窗
            if len(prediction_history) >= 5:
                # 使用刚计算的均值帧
                avg_frame = (prediction_history[-1][0] + prediction_history[-2][1] + 
                           prediction_history[-3][2] + prediction_history[-4][3] + 
                           prediction_history[-5][4]) / 5.0
                new_frame = avg_frame.reshape(1, -1)
            else:
                # 如果还没有足够的历史，使用当前预测的第1帧
                new_frame = ctrl_pred_np[0:1]
        
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