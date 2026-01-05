import torch
import torch.nn as nn
from torchdiffeq import odeint

# Simple ODE function (learnable dynamics)
class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t, h):
        # t is required by odeint, but unused
        return self.net(h)


class LottoHybridModel(nn.Module):
    def __init__(self, pool_size=45, sequence_length=30, hidden_size=128):
        super().__init__()
        self.pool_size = pool_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        # Embed numbers into dense vectors
        self.embedding = nn.Embedding(pool_size + 1, hidden_size)  

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # ODE dynamics
        self.odefunc = ODEFunc(hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, pool_size)

    def forward(self, x):
        # x: (batch, seq_len)
        if not x.dtype == torch.long:
            x = x.long()  # ✅ ensure embedding indices are integers
            
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, hidden)
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Take last hidden state
        h = out[:, -1, :]  

        # Apply ODE integration for temporal dynamics
        t = torch.linspace(0, 1, 2).to(x.device)  # simple time horizon
        h_ode = odeint(self.odefunc, h, t)[-1]    # (batch, hidden)

        # Combine
        h_final = (h + h_ode) / 2  

        logits = self.fc(h_final)  # (batch, pool_size)
        return logits

    def suggest_combination(self, x, temperature=1.0, top_k=6):
        logits = self.forward(x)
        probs = torch.softmax(logits / temperature, dim=-1)
        numbers = torch.multinomial(probs, num_samples=top_k, replacement=False)
        return numbers[0], logits[0]
