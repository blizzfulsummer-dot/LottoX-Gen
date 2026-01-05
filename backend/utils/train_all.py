import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from utils.helpers import prepare_dataset
from models.lotto_ode_model import LottoODEModel

# -----------------------
# Training parameters
# -----------------------
SEQUENCE_LENGTH = 30
EPOCHS = 100
LEARNING_RATE = 0.01

# -----------------------
# Locate all JSON history files in `data/`
# -----------------------
data_dir = Path("data")
history_files = list(data_dir.glob("*.json"))

# -----------------------
# Safe checkpoint loader
# -----------------------
def load_checkpoint_safe(model, weight_file):
    checkpoint = torch.load(weight_file)
    model_dict = model.state_dict()
    # Only load layers with matching shapes
    pretrained_dict = {k: v for k, v in checkpoint.items() 
                       if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    skipped = [k for k in checkpoint if k not in pretrained_dict]
    if skipped:
        print("⚠️ Skipped mismatched layers:", skipped)

# -----------------------
# Training function per variant
# -----------------------
def train_variant(file_path):
    variant = file_path.stem.replace("lotto_history_", "")
    print(f"🔹 Training variant {variant}...")

    # Load history from JSON
    with open(file_path, "r") as f:
        history = json.load(f)

    # Determine pool size from the highest number in history
    pool_size = max(max(draw) for draw in history)
    print(f"Variant {variant} has pool_size={pool_size}")

    # Prepare dataset
    X, Y = prepare_dataset(history, pool_size, SEQUENCE_LENGTH)

    # Initialize model
    model = LottoODEModel(pool_size=pool_size, sequence_length=SEQUENCE_LENGTH)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Check for existing checkpoint
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    weight_file = weights_dir / f"weights_{variant}.pth"
    if weight_file.exists():
        print(f"🔄 Loading existing checkpoint for variant {variant}...")
        load_checkpoint_safe(model, weight_file)

    # Training loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        probs = model(X)
        loss = criterion(probs, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[{variant}] Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

    # Save weights
    torch.save(model.state_dict(), weight_file)
    print(f"✅ Variant {variant} model saved as {weight_file}\n")
    return variant

# -----------------------
# Train all variants in parallel
# -----------------------
with ThreadPoolExecutor(max_workers=len(history_files)) as executor:
    futures = [executor.submit(train_variant, f) for f in history_files]
    for f in futures:
        f.result()
