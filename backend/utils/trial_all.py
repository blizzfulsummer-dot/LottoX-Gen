import torch
import os
from .helpers import draw_to_onehot, compute_features
from models.lotto_ode_model import LottoODEModel

# -----------------------
# Apply temperature to probabilities
# -----------------------
def apply_temperature(probs, temperature=1.0):
    """
    Scales probabilities using a temperature parameter.
    Lower temperature -> sharper distribution.
    Higher temperature -> more uniform distribution.
    """
    probs = probs / temperature
    probs = torch.softmax(probs, dim=-1)
    return probs

# -----------------------
# Suggest a single combination
# -----------------------
def suggest_combination(model, extended_features, draw_size=6, temperature=0.7):
    """
    Generates a suggested combination from model probabilities.
    """
    probs = model.forward(extended_features).squeeze(0)
    probs = apply_temperature(probs, temperature=temperature)

    chosen = []
    available = torch.ones(len(probs), dtype=torch.bool)

    for _ in range(draw_size):
        masked_probs = probs * available.float()
        masked_probs = masked_probs / masked_probs.sum()
        pick = torch.multinomial(masked_probs, 1).item()
        chosen.append(pick + 1)
        available[pick] = False
        probs = probs * (1 + model.co_occurrence[pick])
        probs = probs / probs.sum()

    return sorted(chosen)

# -----------------------
# Trial all variants at multiple temperatures
# -----------------------
def trial_all_multi_temp(lotto_configs, num_trials=50, temperatures=[0.5, 0.7, 1.0]):
    """
    Generates multiple suggestions for all lottery variants at multiple temperatures.
    Returns a nested dictionary: {variant: {temperature: [list of combinations]}}
    """
    all_suggestions = {}

    for variant, config in lotto_configs.items():
        print(f"\nGenerating suggestions for {variant}...")
        pool_size = config["pool_size"]
        sequence_length = config["sequence_length"]
        history = config["history"]
        num_pairs = config.get("num_pairs", 20)

        # Initialize model and load weights
        model = LottoODEModel(pool_size=pool_size, sequence_length=sequence_length, num_pairs=num_pairs)
        weight_file = f"backend/weights_{variant}.pth"
        if not os.path.exists(weight_file):
            print(f"Weight file not found for {variant}, skipping...")
            continue
        model.load_state_dict(torch.load(weight_file))
        model.eval()

        # Prepare latest sequence for prediction
        last_sequence = history[-sequence_length:]
        draw_tensor = torch.cat([draw_to_onehot(d, pool_size) for d in last_sequence])
        freq_vec, co_vec = compute_features(last_sequence, pool_size, num_pairs)
        extended_features = torch.cat([draw_tensor, freq_vec, co_vec]).unsqueeze(0)

        # Generate suggestions for each temperature
        temp_suggestions = {}
        for temp in temperatures:
            suggestions = []
            for _ in range(num_trials):
                combo = suggest_combination(model, extended_features, draw_size=6, temperature=temp)
                suggestions.append(combo)
            temp_suggestions[temp] = suggestions
            print(f"  Temperature {temp}: generated {num_trials} combinations.")

        all_suggestions[variant] = temp_suggestions

    return all_suggestions
