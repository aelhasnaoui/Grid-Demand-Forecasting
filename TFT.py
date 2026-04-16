import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

class GridDataset(Dataset):
    """
    Handles real-world PJM data. 
    Implements a rolling window for 24h, 48h, and 72h horizons.
    """
    def __init__(self, csv_path: str, history_size: int = 168, horizons: list = [24, 48, 72]):
        # Load real data (e.g., PJM_Load_hourly.csv)
        self.df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.data = self.df.values
        self.history_size = history_size
        self.horizons = horizons
        self.scaler = RobustScaler() # Better for outliers in grid data
        self.scaled_data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.scaled_data) - self.history_size - max(self.horizons)

    def __getitem__(self, idx):
        # Input: 1 week of history (168 hours)
        x = self.scaled_data[idx : idx + self.history_size]
        
        # Targets: Multi-horizon slices
        y_24 = self.scaled_data[idx + self.history_size + 23]
        y_48 = self.scaled_data[idx + self.history_size + 47]
        y_72 = self.scaled_data[idx + self.history_size + 71]
        
        return torch.tensor(x, dtype=torch.float32), {
            "24h": torch.tensor(y_24, dtype=torch.float32),
            "48h": torch.tensor(y_48, dtype=torch.float32),
            "72h": torch.tensor(y_72, dtype=torch.float32)
        }


class GridAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: [Batch, Seq, Hidden*2]
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)

class ResidualGridNet(nn.Module):
    """
    Bi-LSTM with Cross-Attention + Multi-Task Output Heads.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Captures temporal dependencies in both directions
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = GridAttention(hidden_dim)
        
        # Multi-task heads for different horizons
        self.h24 = nn.Linear(hidden_dim * 2, input_dim)
        self.h48 = nn.Linear(hidden_dim * 2, input_dim)
        self.h72 = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        
        # Predicting the residual (the change from the last known state)
        last_val = x[:, -1, :] 
        res_24 = self.h24(context)
        res_48 = self.h48(context)
        res_72 = self.h72(context)
        
        return {
            "24h": last_val + res_24, 
            "48h": last_val + res_48, 
            "72h": last_val + res_72
        }


def train_step(model, batch, optimizer, criterion):
    x, targets = batch
    preds = model(x)
    
    # Dynamic weighting: prioritize 24h accuracy for spinning reserves
    # but don't let 72h error explode.
    loss_24 = criterion(preds["24h"], targets["24h"])
    loss_48 = criterion(preds["48h"], targets["48h"])
    loss_72 = criterion(preds["72h"], targets["72h"])
    
    total_loss = (loss_24 * 1.0) + (loss_48 * 0.5) + (loss_72 * 0.2)
    
    total_loss.backward()
    optimizer.step()
    return total_loss.item()