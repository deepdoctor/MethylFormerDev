import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ===================== 你已有的模型代码（原样保留） =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class MethylationTransformer(nn.Module):
    def __init__(self, seq_len, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(MethylationTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        x = x.transpose(1, 2)            # [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)     # [batch, d_model]
        return self.fc_out(x)

# ===================== 模拟数据生成器（含大量 0） =====================
def generate_synthetic_methylation_data(
    num_samples: int,
    seq_len: int,
    zero_block_prob: float = 0.6,
    zero_block_max: int = 80,
    noise_scale: float = 0.05,
    seed: int = 42
):
    """
    生成模拟甲基化序列和对应的“p值”标签。
    - 大量的0通过在序列里插入随机长度的“零块”实现（模拟覆盖度低/未甲基化区域）。
    - 目标p值与中心窗口（模拟TSS附近）甲基化均值负相关（低甲基化 -> 更大可能有CTCF活性 -> 这里给出更小/或更大p值任选）
      这里我们设置：p = sigmoid( a * (mean_center) + b * (global_mean) + noise )，你可以按需要改成相反方向。
    """
    rng = np.random.default_rng(seed)
    X = np.clip(rng.normal(loc=0.5, scale=0.2, size=(num_samples, seq_len)), 0.0, 1.0)

    # 注入随机“零块”，让数据出现较多的0
    for i in range(num_samples):
        # 以一定概率插入1~N个零块
        if rng.random() < zero_block_prob:
            n_blocks = rng.integers(1, 4)
            for _ in range(n_blocks):
                block_len = int(rng.integers(10, zero_block_max))
                start = int(rng.integers(0, max(1, seq_len - block_len)))
                X[i, start:start+block_len] = 0.0

    # 计算目标（模拟）
    center = seq_len // 2
    half_w = 25  # 中心窗口大小，可理解为TSS±25 bin
    center_mean = X[:, max(0, center - half_w):min(seq_len, center + half_w)].mean(axis=1)
    global_mean = X.mean(axis=1)

    # 生成一个可学习关系：p ~ sigmoid(a * center_mean + b * global_mean + noise)
    a, b = 3.0, 1.5
    noise = rng.normal(0, noise_scale, size=(num_samples,))
    logits = a * center_mean + b * global_mean + noise
    p_vals = 1 / (1 + np.exp(-logits))  # (0,1)

    # 转成张量并加上最后一维（input_dim=1）
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)        # [N, L, 1]
    y = torch.tensor(p_vals, dtype=torch.float32).unsqueeze(-1)   # [N, 1]
    return X, y

# ===================== Dataset 定义 =====================
class MethylCTCFDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        super().__init__()
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===================== 训练与验证流程 =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)

            # 收集预测值和真实值
            preds_list.append(pred.cpu().numpy())
            targets_list.append(yb.cpu().numpy())

    # 合并所有 batch
    preds_all = np.concatenate(preds_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)

    # 计算相关性（逐列相关性可以改写）
    if preds_all.ndim == 1 or preds_all.shape[1] == 1:
        # 一维情况
        corr = pearsonr(preds_all.ravel(), targets_all.ravel())[0]
    else:
        # 多维输出情况：逐列相关性
        corr = np.array([pearsonr(preds_all[:, i], targets_all[:, i])[0] 
                         for i in range(preds_all.shape[1])])

    mean_loss = total_loss / len(loader.dataset)
    return mean_loss, corr, preds_all, targets_all

# ===================== 主程序：生成数据 + 划分 + 训练/验证 =====================
def main():
    # 配置
    result_dir = "/media/desk16/zhiwei/paper_code/CpGPT/tutorials/weights/"
    seq_len = 2000
    input_dim = 1
    num_samples = 8000
    batch_size = 64
    epochs = 5
    lr = 1e-4
    val_ratio = 0.2
    test_ratio = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("the device is", device)

    # 读取数据
    X, y = load_data()

    # 划分训练 / 验证 / 测试
    N = X.shape[0]
    idx = torch.randperm(N)
    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Dataset / DataLoader
    train_ds = MethylCTCFDataset(X_train, y_train)
    val_ds   = MethylCTCFDataset(X_val, y_val)
    test_ds  = MethylCTCFDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    # 模型
    model = MethylationTransformer(seq_len=seq_len, input_dim=input_dim, d_model=128, nhead=8, num_layers=4)
    model.to(device)

    model.load_state_dict(torch.load(result_dir + "best_methyl_transformer.pt", map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, corr, preds_all, targets_all = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), result_dir + "best_methyl_transformer.pt")

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train MSE: {train_loss:.4f} | "
              f"Val MSE: {val_loss:.4f} | "
              f"Best Val: {best_val:.4f} | "
              f"corr Val: {corr:.4f} | ")

    print("Training done. Best checkpoint saved.")

    # ===== 测试集评估 =====
    model.load_state_dict(torch.load(result_dir + "best_methyl_transformer.pt", map_location=device))
    test_loss, test_corr, preds_test, targets_test = evaluate(model, test_loader, criterion, device)
    print(f"Test MSE: {test_loss:.4f} | corr Test: {test_corr:.4f}")

    np.save(result_dir + "preds_test.npy", preds_test)
    np.save(result_dir + "targets_test.npy", targets_test)

def load_data():
    import numpy as np
    import pandas as pd
    data_dir = "/media/desk16/zhiwei/paper_code/CpGPT/tutorials/processedData/"
    data = pd.read_csv(data_dir + "E113.csv")
    CTCF_val = np.load(data_dir + "CTCF_values.npy")
    col_means = np.mean(CTCF_val, axis=1)
    col_median = np.median(CTCF_val, axis=1)
    meth_values = np.load(data_dir + "meth_values.npy")
    X = torch.tensor(meth_values, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(col_means, dtype=torch.float32).unsqueeze(-1)
    return X,y
    # return X[:1000,:,:],y[:1000,:]



if __name__ == "__main__":
    main()
