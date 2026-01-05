import json
from collections import Counter
from pathlib import Path

class BayesianMemory:
    """
    Keeps track of number frequencies and updates Bayesian priors dynamically.
    """

    def __init__(self, variant: str, data_dir: Path = Path("data")):
        self.variant = variant
        self.data_dir = data_dir
        self.memory_file = data_dir / f"bayesian_memory_{variant}.json"
        self.freq = Counter()
        self.total = 0
        self.load()

    def load(self):
        if self.memory_file.exists():
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                self.freq = Counter(data["freq"])
                self.total = data["total"]

    def update(self, new_draw: list[int]):
        """Add new lotto draw to the frequency memory."""
        self.freq.update(new_draw)
        self.total += len(new_draw)
        self.save()

    def save(self):
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "w") as f:
            json.dump({"freq": self.freq, "total": self.total}, f, indent=2)

    def get_prior_probs(self, pool_size: int):
        """Return normalized prior probabilities from frequency counts."""
        probs = [self.freq.get(i + 1, 0) for i in range(pool_size)]
        s = sum(probs)
        if s == 0:
            return [1.0 / pool_size] * pool_size
        return [p / s for p in probs]
