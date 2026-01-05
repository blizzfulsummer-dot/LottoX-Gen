import torch
from itertools import combinations

def draw_to_onehot(draw, pool_size):
    vec = torch.zeros(pool_size)
    for n in draw:
        vec[n-1] = 1
    return vec

def compute_features(sequence, pool_size, num_pairs=20, num_triples=10):
    """
    Compute additional features from a sequence of draws:
    - Frequency of each number
    - Top `num_pairs` co-occurring pairs
    - Top `num_triples` co-occurring triples
    """
    # --- Frequency ---
    freq = torch.zeros(pool_size)
    for draw in sequence:
        for n in draw:
            freq[n - 1] += 1
    freq = freq / len(sequence)  # normalize

    # --- Pair co-occurrence ---
    pair_counts = {}
    for draw in sequence:
        for a, b in combinations(draw, 2):
            key = tuple(sorted((a, b)))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    # Take top `num_pairs` pairs
    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:num_pairs]
    co_vec = torch.zeros(num_pairs)
    for i, ((a, b), count) in enumerate(top_pairs):
        co_vec[i] = count / len(sequence)  # normalize

    # --- Triple co-occurrence ---
    triple_counts = {}
    for draw in sequence:
        for a, b, c in combinations(draw, 3):
            key = tuple(sorted((a, b, c)))
            triple_counts[key] = triple_counts.get(key, 0) + 1

    # Take top `num_triples` triples
    top_triples = sorted(triple_counts.items(), key=lambda x: x[1], reverse=True)[:num_triples]
    triple_vec = torch.zeros(num_triples)
    for i, ((a, b, c), count) in enumerate(top_triples):
        triple_vec[i] = count / len(sequence)  # normalize

    return freq, co_vec, triple_vec

def prepare_dataset(history, pool_size, sequence_length=30, num_pairs=20, num_triples=10):
    X_seq, Y_seq = [], []
    if len(history) <= sequence_length:
        seq = history[-sequence_length:]
        target = history[-1]  # last draw as target
        draw_tensor = torch.cat([draw_to_onehot(d, pool_size) for d in seq])
        freq_vec, co_vec, triple_vec = compute_features(seq, pool_size, num_pairs, num_triples)
        extended_features = torch.cat([draw_tensor, freq_vec, co_vec, triple_vec])
        X_seq.append(extended_features)
        Y_seq.append(draw_to_onehot(target, pool_size))
    else:
        for i in range(len(history) - sequence_length):
            seq = history[i:i+sequence_length]
            target = history[i+sequence_length]
            draw_tensor = torch.cat([draw_to_onehot(d, pool_size) for d in seq])
            freq_vec, co_vec, triple_vec = compute_features(seq, pool_size, num_pairs, num_triples)
            extended_features = torch.cat([draw_tensor, freq_vec, co_vec, triple_vec])
            X_seq.append(extended_features)
            Y_seq.append(draw_to_onehot(target, pool_size))

    X = torch.stack(X_seq)
    Y = torch.stack(Y_seq)
    return X, Y

