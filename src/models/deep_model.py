"""Deep learning model: small LSTM with regularisation for small-data forecasting."""
import torch
import torch.nn as nn


class FoodInflationLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                             dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
