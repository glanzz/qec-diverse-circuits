import os
import json
import numpy as np


class Utility:

    @classmethod
    def safe_normalize(cls, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @classmethod
    def get_random_pi_value(cls):
        n = np.random.randint(-100, 100)
        if n == 0:
            return np.pi / 1000
        return np.pi / n

    @classmethod
    def save_json(cls, file_path, data):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def create_next_versioned_directory(cls, base_path, prefix="v"):
        os.makedirs(base_path, exist_ok=True)
        existing_dirs = [
            d
            for d in os.listdir(base_path)
            if d.startswith(prefix) and d[len(prefix):].isdigit()
        ]
        existing_numbers = sorted(int(d[len(prefix):]) for d in existing_dirs)
        next_number = existing_numbers[-1] + 1 if existing_numbers else 0
        new_dir_path = os.path.join(base_path, f"{prefix}{next_number}")
        os.makedirs(new_dir_path, exist_ok=True)
        return new_dir_path

    @classmethod
    def get_latest_versioned_directory(cls, base_path, prefix="v"):
        if not os.path.exists(base_path):
            return None
        version_dirs = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and d.startswith(prefix)
        ]
        if not version_dirs:
            return None
        latest = max(version_dirs, key=lambda d: int(d[len(prefix):]))
        return os.path.join(base_path, latest)
