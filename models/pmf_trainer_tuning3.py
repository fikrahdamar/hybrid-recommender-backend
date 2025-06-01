import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class PMFTrainer3:
    def __init__(self, n_users, n_items, n_factors=256, lr=0.005, reg=0.02, epochs=20):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        self.user_factors = np.random.normal(scale=0.1, size=(n_users, n_factors))  
        self.losses = []
        self.val_losses = []

    def load_item_factors(self, encoded_amazon_vectors_path, mapping_path):
        df_qi_raw = pd.read_csv(encoded_amazon_vectors_path, dtype=str).apply(pd.to_numeric, errors="coerce")
        df_mapping = pd.read_csv(mapping_path)

        df_qi_raw.set_index("asin", inplace=True)
        q_i = np.zeros((self.n_items, self.n_factors))
        missing_count = 0
        valid_items = set()

        for _, row in df_mapping.iterrows():
            item_idx = int(row["item_index"])
            asin = row["asin"]

            if asin in df_qi_raw.index:
                q_i[item_idx] = df_qi_raw.loc[asin].values
                valid_items.add(item_idx)
            else:
                # Jangan isi vector random — kita anggap saja tidak ada
                missing_count += 1

        print(f"Missing ASIN vectors: {missing_count}")
        return q_i, valid_items


    def predict_all(self, df, q_i, output_path="prediction.csv"):
        """
        Simpan prediksi model untuk semua pasangan user-item yang tersedia di df.
        """
        result = []
        for _, row in df.iterrows():
            u = int(row["user_index"])
            i = int(row["item_index"])
            r_ui = float(row["rating"])
            pred = np.dot(self.user_factors[u], q_i[i])
            result.append({
                "user_index": u,
                "item_index": i,
                "true_rating": r_ui,
                "predicted_rating": pred
            })

        pred_df = pd.DataFrame(result)
        pred_df.to_csv(output_path, index=False)
        print(f"Prediction results saved to {output_path}")
    
    
    def evaluate_rmse(self, df, q_i):
        errors = []
        for _, row in df.iterrows():
            u = int(row["user_index"])
            i = int(row["item_index"])
            r_ui = float(row["rating"])

            pred = np.dot(self.user_factors[u], q_i[i])
            pred = np.clip(pred, 0, 1)
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