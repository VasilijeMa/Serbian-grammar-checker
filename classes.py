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
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordBetweenModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_before, x_after):
        embed_before = self.embedding(x_before)
        # print(f"Before {embed_before}")
        embed_after = self.embedding(x_after)
        
        combined = torch.cat((embed_before, embed_after), dim=2)
        combined = combined.squeeze(1)
        lstm_out, _ = self.lstm(combined)
        
        out = self.fc(lstm_out[:, -1, :])
        return out