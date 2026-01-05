import torch
import torch.nn as nn
from torchdiffeq import odeint

# ---------- Neural ODE ----------
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )
    def forward(self, t, z):
        return self.fc(z)

# ---------- Expert & Router ----------
class Expert(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, z):
        return self.fc(z)

class Router(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.fc = nn.Linear(dim, K)
    def forward(self, z, k=2):
        logits = self.fc(z)
        topk_vals, topk_idx = torch.topk(logits, k)
        mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0)
        big_neg = -1e9 * (1 - mask)
        sparse_logits = logits + big_neg
        alpha = torch.softmax(sparse_logits, dim=-1)
        return alpha

# ---------- Lotto Neural ODE with Multi-Pattern ----------
class LottoODEModel(nn.Module):
    def __init__(self, pool_size, draw_size=6, latent_dim=32, K=4, k=2,
                 sequence_length=30, num_pairs=20, num_triples=10):
        super().__init__()
        self.pool_size = pool_size
        self.draw_size = draw_size
        self.sequence_length = sequence_length
        self.num_pairs = num_pairs
        self.num_triples = num_triples
        self.input_dim = pool_size * sequence_length + pool_size + num_pairs + num_triples

        # Encoder
        self.encoder = nn.Linear(self.input_dim, latent_dim)

        # Multi-pattern ODEs
        self.ode_presence = ODEFunc(latent_dim)
        self.ode_frequency = ODEFunc(latent_dim)
        self.ode_co = ODEFunc(latent_dim)

        # Experts & Router
        self.router = Router(latent_dim, K)
        self.experts = nn.ModuleList([Expert(latent_dim, pool_size) for _ in range(K)])
        self.k = k

        # Learnable co-occurrence matrix
        self.co_occurrence = nn.Parameter(torch.eye(pool_size))

    # ---------- Forward pass ----------
    def forward(self, draw_history):
        z0 = self.encoder(draw_history)
        t = torch.tensor([0.0, 1.0])

        z_presence = odeint(self.ode_presence, z0, t)[-1]
        z_freq = odeint(self.ode_frequency, z0, t)[-1]
        z_co = odeint(self.ode_co, z0, t)[-1]

        def expert_out(z):
            alpha = self.router(z, self.k)
            expert_outs = torch.stack([e(z) for e in self.experts], dim=1)
            out = (alpha.unsqueeze(-1) * expert_outs).sum(1)
            return torch.sigmoid(out)

        prob_presence = expert_out(z_presence)
        prob_freq = expert_out(z_freq)
        prob_co = expert_out(z_co)

        combined_prob = 0.4*prob_presence + 0.3*prob_freq + 0.3*prob_co
        combined_prob = torch.sigmoid(combined_prob @ torch.sigmoid(self.co_occurrence))
        combined_prob = combined_prob / combined_prob.sum()  # normalize
        return combined_prob

    # ---------- Temperature ----------
    def apply_temperature(self, prob, temperature=1.0):
        p_temp = prob ** (1 / temperature)
        p_temp = p_temp / p_temp.sum()
        return p_temp

    # ---------- Suggest combination ----------
    def suggest_combination(self, draw_history, temperature=1.0):
        probs = self.forward(draw_history).squeeze(0)

        # --- Safety checks ---
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs, dtype=torch.float32)

        # Replace NaN/Inf with zeros, clamp negative values
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)

        # Ensure non-zero sum for normalization
        if probs.sum() == 0:
            probs = torch.ones_like(probs) / probs.numel()

        chosen = []
        available = torch.ones(self.pool_size, dtype=torch.bool)

        for _ in range(self.draw_size):
            masked_probs = probs * available.float()
            masked_probs = self.apply_temperature(masked_probs, temperature)

            if masked_probs.sum() == 0:
                masked_probs = torch.ones_like(masked_probs) / masked_probs.numel()

            pick = torch.multinomial(masked_probs, 1).item()
            chosen.append(pick + 1)
            available[pick] = False

            # Update probabilities based on co-occurrence
            probs = probs * (1 + self.co_occurrence[pick])
            probs = probs / probs.sum()

        return sorted(chosen), probs.detach().numpy()