import torch
from collections import Counter
import numpy as np

def bayesian_smooth(ai_probs, history, alpha: float = 100.0):
    """
    Blend neural AI probabilities with empirical frequencies using Dirichlet Bayesian smoothing.
    
    ai_probs : torch.Tensor or list[float]
        Probabilities predicted by the AI model.
    history : list[list[int]]
        Past lotto draws (each is a list of numbers).
    alpha : float
        Strength of AI confidence vs. historical data.
        Higher alpha = rely more on AI model.
    """

    if not isinstance(ai_probs, torch.Tensor):
        ai_probs = torch.tensor(ai_probs, dtype=torch.float32)

    pool_size = ai_probs.numel()

    # Flatten historical draws into counts
    flat = [n for draw in history for n in draw]
    counts = Counter(flat)
    
    # Convert counts to frequency vector
    history_freq = torch.zeros(pool_size, dtype=torch.float32)
    for num, count in counts.items():
        if 1 <= num <= pool_size:
            history_freq[num - 1] = count

    # Normalize to probability distribution
    history_probs = history_freq / (history_freq.sum() + 1e-9)

    # Bayesian posterior update (Dirichlet-like smoothing)
    blended = (alpha * ai_probs + history_probs * history_freq.sum()) / (alpha + history_freq.sum())
    blended /= blended.sum()  # ensure normalization

    return blended
