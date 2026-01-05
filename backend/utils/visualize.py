import torch
import matplotlib.pyplot as plt
from .helpers import draw_to_onehot, compute_features

def visualize_probabilities(model, draw_history, pool_size, num_pairs=20, num_triples=10, temperatures=[0.5, 0.7, 1.0]):
    """
    Visualizes predicted probabilities for different temperature values.
    draw_history: list of past draws (list of lists)
    """
    # Flatten draw history into one-hot
    draw_tensor = torch.cat([draw_to_onehot(d, pool_size) for d in draw_history])

    # Compute additional features
    freq_vec, co_vec, triple_vec = compute_features(draw_history, pool_size, num_pairs, num_triples)

    # Combine into extended feature tensor
    extended_tensor = torch.cat([draw_tensor, freq_vec, co_vec, triple_vec]).unsqueeze(0)

    # Plot probabilities for each temperature
    plt.figure(figsize=(12, 5))
    for temp in temperatures:
        probs = model.apply_temperature(model(extended_tensor).squeeze(0), temp).detach().numpy()
        plt.plot(range(1, pool_size + 1), probs, marker='o', label=f'T={temp}')
    
    plt.xlabel("Number")
    plt.ylabel("Probability")
    plt.title("Predicted Number Probabilities with Different Temperatures")
    plt.xticks(range(1, pool_size + 1))
    plt.grid(True)
    plt.legend()
    plt.show()
