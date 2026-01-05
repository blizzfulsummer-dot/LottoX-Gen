import torch
import torch.nn as nn
import torch.optim as optim

from utils.helpers import draw_to_onehot, compute_features, prepare_dataset
from models.lotto_ode_model import LottoODEModel
from utils.visualize import visualize_probabilities

# -----------------------
# Configuration
# -----------------------
pool_size = 42
draw_size = 6
sequence_length = 4
num_pairs = 20
num_triples = 10
epochs = 10
lr = 0.01

# -----------------------
# Example history (replace with your real draws)
# -----------------------
history = [
    [5, 12, 18, 23, 32, 41],
    [3, 7, 19, 28, 35, 39],
    [1, 9, 12, 19, 28, 40],
    [2, 15, 22, 29, 33, 41],
    [4, 11, 18, 27, 36, 42],
    [2, 9, 16, 23, 35, 38],
    # add more draws as needed
]

# -----------------------
# Prepare dataset
# -----------------------
X, Y = prepare_dataset(history, pool_size, sequence_length, num_pairs, num_triples)

# -----------------------
# Initialize model, loss, optimizer
# -----------------------
model = LottoODEModel(pool_size=pool_size, draw_size=draw_size, sequence_length=sequence_length)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------
# Training loop
# -----------------------
print(f"🔹 Training test model for {epochs} epochs...")
for epoch in range(epochs):
    optimizer.zero_grad()
    probs = model(X)
    loss = criterion(probs, Y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# -----------------------
# Suggest a combination from the last sequence
# -----------------------
last_seq = history[-sequence_length:]
draw_tensor = torch.cat([draw_to_onehot(d, pool_size) for d in last_seq])
freq_vec, co_vec, triple_vec = compute_features(last_seq, pool_size, num_pairs, num_triples)
extended_tensor = torch.cat([draw_tensor, freq_vec, co_vec, triple_vec]).unsqueeze(0)

suggested, probs = model.suggest_combination(extended_tensor)
print("🔹 Suggested combination:", suggested)

# -----------------------
# Visualize probabilities (optional)
# -----------------------
visualize_probabilities(model, last_seq, pool_size=pool_size, num_pairs=num_pairs, num_triples=num_triples)
