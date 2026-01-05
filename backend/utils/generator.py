import random
import json

def generate_dummy_history(pool_size, num_draws=40, draw_size=6):
    history = []
    for _ in range(num_draws):
        draw = sorted(random.sample(range(1, pool_size+1), draw_size))
        history.append(draw)
    return history

variants = {
    "6_42": 42,
    "6_45": 45,
    "6_49": 49,
    "6_55": 55,
    "6_58": 58,
}

for variant, pool_size in variants.items():
    history = generate_dummy_history(pool_size)
    with open(f"data/{variant}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"{variant}: {len(history)} draws saved")
