from fastapi import FastAPI, Body, HTTPException
from pathlib import Path
import torch
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from models.lotto_ode_model import LottoODEModel
from models.lotto_hybrid_model import LottoHybridModel
from utils.helpers import prepare_dataset
from utils.bayesian_layer import bayesian_smooth
from utils.bayesian_memory import BayesianMemory
from typing import Union

app = FastAPI()

# Serve frontend
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

    # Fallback: if hybrid history not found, use base variant history
    if not data_file.exists() and variant.endswith("_hybrid"):
        base_variant = variant.replace("_hybrid", "")
        data_file = Path("data") / f"lotto_history_{base_variant}.json"

    if not data_file.exists():
        print(f"⚠️ Historical data for variant {variant} not found, skipping.")
        continue

    with open(data_file) as f:
        history = json.load(f)

    pool_size = max(max(draw) for draw in history)

    # Pick model class depending on variant
    if variant.endswith("_hybrid"):
        from models.lotto_hybrid_model import LottoHybridModel
        model = LottoHybridModel(pool_size=pool_size, sequence_length=30)
    else:
        from models.lotto_ode_model import LottoODEModel
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
    logits = torch.clamp(logits, min=-50.0, max=50.0)

    try:
        max_val = logits.max()
        exp_logits = torch.exp(logits - max_val)
        probs = exp_logits / exp_logits.sum()
        if not torch.isfinite(probs).all():
            probs = torch.ones_like(probs) / probs.numel()
    except:
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
        return {"error": f"Invalid mode. Choose from {list(mode_map.keys())!r}"}

    temperature = mode_map[mode]

    if variant not in model_cache:
        return {"error": f"Variant {variant} not found."}

    model = model_cache[variant]

    # Load correct history file
    data_file = Path("data") / f"lotto_history_{variant}.json"
    if not data_file.exists() and variant.endswith("_hybrid"):
        base_variant = variant.replace("_hybrid", "")
        data_file = Path("data") / f"lotto_history_{base_variant}.json"

    if not data_file.exists():
        return {"error": "Historical data not found."}

    with open(data_file) as f:
        history = json.load(f)

    sequence_length = 30
    last_seq = history[-sequence_length:]

    # Padding for short histories
    if len(last_seq) < sequence_length:
        pad_count = sequence_length - len(last_seq)
        last_seq = [[1]*model.pool_size]*pad_count + last_seq

    # Clip numbers to valid range
    last_seq_clipped = [[min(max(1, n), model.pool_size) for n in draw] for draw in last_seq]

    if not last_seq_clipped:
        last_seq_clipped = [[1]*model.pool_size]*sequence_length

    try:
        X, _ = prepare_dataset(last_seq_clipped, pool_size=model.pool_size, sequence_length=sequence_length)
        input_tensor = X[-1].unsqueeze(0).clone()

        # Auto-detect model type
        if isinstance(model, LottoHybridModel):
            input_tensor = input_tensor.to(torch.long)
        elif isinstance(model, LottoODEModel):
            input_tensor = input_tensor.to(torch.float32)

        with torch.no_grad():
            suggested, logits = model.suggest_combination(input_tensor, temperature=temperature)

            # Convert logits to tensor if it's a NumPy array
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            # Hybrid model: generate unique numbers
            if isinstance(model, LottoHybridModel):
                combination_size = suggested.shape[-1] if isinstance(suggested, torch.Tensor) else len(suggested)
                chosen = []
                logits_copy = logits.clone()
                for _ in range(combination_size):
                    probs = stable_softmax(logits_copy, temperature=temperature)
                    probs = torch.clamp(probs, min=1e-8)
                    probs /= probs.sum()
                    idx = torch.multinomial(probs, 1).item()
                    chosen.append(idx + 1)  # +1 because lottery numbers start at 1
                    #logits_copy[idx] = -float('inf')  # mask to prevent duplicates
                    logits_copy[idx] = -1e9
                suggested = torch.tensor(chosen, dtype=torch.int64)

            # ODE model: ensure tensor
            if not isinstance(suggested, torch.Tensor):
                suggested = torch.tensor(suggested, dtype=torch.int64)

            # Stable softmax for probabilities
            probs = stable_softmax(logits, temperature=temperature)
            # Bayesian post-processing (empirical correction)
            probs = bayesian_smooth(probs, history, alpha=80.0)

        return {
            "mode": mode,
            "temperature": temperature,
            "suggested": suggested.tolist(),
            "probabilities": probs.tolist()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "retry", "message": "AI is temporarily busy/confused. Try again or switch mode."}



@app.post("/update/{variant}")
def update_history(
    variant: str,
    draws: Union[list[int], list[list[int]]] = Body(...)
):
    """Append one or more validated lottery draws to history."""

    data_file = Path("data") / f"lotto_history_{variant}.json"

    if not data_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No history found for variant: {variant}"
        )

    # Normalize a single draw into a list of draws.
    if not draws:
        raise HTTPException(status_code=400, detail="No draws provided.")

    if isinstance(draws[0], list):
        new_draws = draws
    else:
        new_draws = [draws]

    with open(data_file, "r") as f:
        history = json.load(f)

    if not isinstance(history, list) or not history:
        raise HTTPException(
            status_code=400,
            detail="History is empty or has an invalid format."
        )

    # Derive basic constraints from existing history.
    expected_count = len(history[0])
    pool_size = max(max(draw) for draw in history)

    added = []
    skipped = []

    for draw in new_draws:
        if not isinstance(draw, list) or len(draw) != expected_count:
            raise HTTPException(
                status_code=400,
                detail=f"Each draw must contain {expected_count} numbers."
            )

        if any(
            not isinstance(n, int) or isinstance(n, bool)
            or n < 1 or n > pool_size
            for n in draw
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Numbers must be integers between 1 and {pool_size}."
            )

        if len(set(draw)) != len(draw):
            raise HTTPException(
                status_code=400,
                detail="Duplicate numbers within a draw are not allowed."
            )

        # Sort for consistent comparison/storage.
        normalized = sorted(draw)

        if normalized in [sorted(old) for old in history + added]:
            skipped.append(normalized)
            continue

        added.append(normalized)

    if added:
        history.extend(added)

        with open(data_file, "w") as f:
            json.dump(history, f, indent=2)

        memory = BayesianMemory(variant)
        for draw in added:
            memory.update(draw)

    return {
        "status": "ok",
        "added": len(added),
        "skipped_duplicates": len(skipped),
        "message": f"Added {len(added)} draw(s); skipped {len(skipped)} duplicate(s)."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
