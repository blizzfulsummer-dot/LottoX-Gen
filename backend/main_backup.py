from fastapi import FastAPI
from pathlib import Path
import torch
import json
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from models.lotto_ode_model import LottoODEModel
from utils.helpers import prepare_dataset

app = FastAPI()


# Serve frontend from the project root's static folder
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/frontend", StaticFiles(directory=frontend_path, html=True), name="frontend")

@app.get("/")
def root():
    return RedirectResponse(url="/frontend/index.html")

# -----------------------
# Load models safely
# -----------------------
weights_dir = Path("weights")
model_cache = {}

for weight_file in weights_dir.glob("*.pth"):
    variant = weight_file.stem.replace("weights_", "")
    data_file = Path("data") / f"lotto_history_{variant}.json"

    if not data_file.exists():
        print(f"⚠️ Historical data for variant {variant} not found, skipping.")
        continue

    with open(data_file) as f:
        history = json.load(f)

    pool_size = max(max(draw) for draw in history)
    model = LottoODEModel(pool_size=pool_size, sequence_length=30)

    checkpoint = torch.load(weight_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    skipped = [k for k in checkpoint if k not in pretrained_dict]
    if skipped:
        print(f"⚠️ Variant {variant} skipped layers due to size mismatch:", skipped)

    model.eval()
    model_cache[variant] = model
    print(f"✅ Variant {variant} loaded successfully with pool_size={pool_size}")


# -----------------------
# Stable softmax with temperature
# -----------------------
def stable_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(temperature, 1e-6)
    logits = logits / temperature

    # Clamp to prevent overflow
    logits = torch.clamp(logits, min=-50.0, max=50.0)

    try:
        max_val = logits.max()
        exp_logits = torch.exp(logits - max_val)
        probs = exp_logits / exp_logits.sum()
        # If still invalid, fallback
        if not torch.isfinite(probs).all():
            probs = torch.ones_like(probs) / probs.numel()
    except:
        # Absolute fallback to uniform
        probs = torch.ones_like(logits) / logits.numel()

    return probs


# -----------------------
# Endpoint: generate lotto suggestion
# -----------------------
@app.post("/suggest/{variant}/{mode}")
def suggest(variant: str, mode: str = "balance"):
    mode_map = {
        "guided-random": 0.5,
        "balance": 0.7,
        "deterministic": 1.0
    }

    if mode not in mode_map:
        return {"error": f"Invalid mode. Choose from {list(mode_map.keys())}"}

    temperature = mode_map[mode]

    if variant not in model_cache:
        return {"error": f"Variant {variant} not found."}

    model = model_cache[variant]
    data_file = Path("data") / f"lotto_history_{variant}.json"

    if not data_file.exists():
        return {"error": "Historical data not found."}

    with open(data_file) as f:
        history = json.load(f)

    sequence_length = 30
    last_seq = history[-sequence_length:]

    # Safe padding if history is too short
    if len(last_seq) < sequence_length:
        pad_count = sequence_length - len(last_seq)
        last_seq = [[1]*model.pool_size]*pad_count + last_seq

    # Clip invalid numbers
    last_seq_clipped = [[min(max(1, n), model.pool_size) for n in draw] for draw in last_seq]

    if not last_seq_clipped:
        last_seq_clipped = [[1]*model.pool_size]*sequence_length

    try:
        X, _ = prepare_dataset(last_seq_clipped, pool_size=model.pool_size, sequence_length=sequence_length)

        with torch.no_grad():
            input_tensor = X[-1].unsqueeze(0).clone()
            suggested, logits = model.suggest_combination(input_tensor, temperature=temperature)

        # Ensure outputs are tensors
        if not isinstance(suggested, torch.Tensor):
            suggested = torch.tensor(suggested, dtype=torch.int64)
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)

        # Apply stable softmax
        probs = stable_softmax(logits, temperature=temperature)

        return {
            "mode": mode,
            "temperature": temperature,
            "suggested": suggested.tolist(),
            "probabilities": probs.tolist()
        }

    except Exception as e:
        print("Model prediction failed: ",str(e))
        return {"status": "retry",
        "message": "AI is temporarily busy/confused. Try again or switch mode."}