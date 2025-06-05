import numpy as np
import pandas as pd
import heapq
import os

class HybridRecommender:
    def __init__(self, user_factors_path, item_factors_path, item_mapping_path):
        self.p_u = np.load(user_factors_path)
        print(f"[INFO] Loaded user factors: {self.p_u.shape}")
        self.q_i = np.load(item_factors_path)
        print(f"[INFO] Loaded item factors: {self.q_i.shape}")
        self.item_mapping = pd.read_csv(item_mapping_path)
        print(f"[INFO] Loaded item mapping: {self.item_mapping.shape[0]} entries")

        
        self.index_to_asin = dict(zip(self.item_mapping['item_index'], self.item_mapping['asin']))

    def recommend(self, user_index, top_k=10):
        if user_index >= len(self.p_u):
            raise ValueError(f"[ERROR] user_index {user_index} di luar jangkauan p_u")

        scores = np.dot(self.q_i, self.p_u[user_index])
        top_indices = heapq.nlargest(top_k, range(len(scores)), scores.__getitem__)
        recommendations = [
            {
                "rank": i+1,
                "item_index": idx,
                "asin": self.index_to_asin.get(idx, "UNKNOWN"),
                "score": float(scores[idx])
            }
            for i, idx in enumerate(top_indices)
        ]

        return recommendations
