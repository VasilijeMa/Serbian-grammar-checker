import torch
import torch.nn as nn
from torch.utils.data import Dataset

class WordDataset(Dataset):
    def __init__(self, X_before, X_after, y):
        self.X_before = X_before
        self.X_after = X_after
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_before[idx], self.X_after[idx], self.y[idx]

class WordBetweenModel(nn.Module):
    def __init__(self, hidden_dim):
        super(WordBetweenModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True).double()
        self.fc = nn.Linear(hidden_dim, 1).double()

    def forward(self, x_before, x_after):
        x_before = x_before.unsqueeze(-1)
        x_after = x_after.unsqueeze(-1)

        combined = torch.cat((x_before, x_after), dim=1)
        combined = combined.unsqueeze(1)

        lstm_out, _ = self.lstm(combined)
        return self.fc(lstm_out[:, -1, :]).squeeze()