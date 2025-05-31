import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class PMFTrainer:
    def __init__(self, n_users, n_items, n_factors=256, lr=0.005, reg=0.02, epochs=20):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        self.user_factors = np.random.normal(scale=0.1, size=(n_users, n_factors))  # p_u
        self.losses = []
        self.val_losses = []

    def load_item_factors(self, encoded_amazon_vectors_path):
        df_qi = pd.read_csv(encoded_amazon_vectors_path)
        asin_vectors = df_qi.drop(columns=["asin"]).values
        return asin_vectors  # q_i

    def evaluate_rmse(self, df, q_i):
        errors = []
        for _, row in df.iterrows():
            u = int(row["user_index"])
            i = int(row["item_index"])
            r_ui = float(row["rating"])

            pred = np.dot(self.user_factors[u], q_i[i])
            errors.append((r_ui - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def fit(self, train_df, q_i, val_df=None):
        for epoch in range(self.epochs):
            total_loss = 0
            for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch+1}"):
                u = int(row["user_index"])
                i = int(row["item_index"])
                r_ui = float(row["rating"])

                p_u = self.user_factors[u]
                q_i_vec = q_i[i]

                pred = np.dot(p_u, q_i_vec)
                error = r_ui - pred

                self.user_factors[u] += self.lr * (error * q_i_vec - self.reg * p_u)

                total_loss += error ** 2 + self.reg * (np.linalg.norm(p_u) ** 2)

            self.losses.append(total_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {total_loss:.4f}")

            if val_df is not None:
                val_rmse = self.evaluate_rmse(val_df, q_i)
                self.val_losses.append(val_rmse)
                print(f"Validation RMSE: {val_rmse:.4f}")

    def save_user_factors(self, path="user_factors.npy"):
        np.save(path, self.user_factors)
        print(f"User factors saved to {path}")

    def plot_losses(self, path=None):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.losses) + 1), self.losses, marker='o', label='Train Loss')
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, marker='x', label='Validation RMSE')
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / RMSE")
        plt.grid(True)
        plt.legend()
        if path:
            plt.savefig(path)
            print(f"Loss plot saved to {path}")
        else:
            plt.show()
