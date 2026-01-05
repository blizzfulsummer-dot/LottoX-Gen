import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from utils.helpers import prepare_dataset
from models.lotto_hybrid_model import LottoHybridModel

def train_hybrid(variant="6_45", epochs=50, lr=1e-3, batch_size=32):
    data_file = Path("data") / f"lotto_history_{variant}.json"
    with open(data_file) as f:
        history = json.load(f)

    pool_size = max(max(draw) for draw in history)
    sequence_length = 30

    # ✅ Call helpers properly
    X, y = prepare_dataset(history, pool_size, sequence_length)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LottoHybridModel(pool_size=pool_size, sequence_length=sequence_length)
    
    # ✅ Stable multi-label loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            # 🔧make sure indices are integers
            batch_x = batch_x.long()

            logits = model(batch_x)
            loss = criterion(logits, batch_y.float())  # ✅ Convert targets to float
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Save weights
    save_path = Path("weights") / f"weights_{variant}_hybrid.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved hybrid model for {variant} at {save_path}")

if __name__ == "__main__":
    for variant in ["6_42", "6_45", "6_49", "6_55", "6_58"]:
        train_hybrid(variant, epochs=50)
