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

    def load_item_factors(self, encoded_amazon_vectors_path, mapping_path, target_std=1.5):
        # Load encoded item vectors
        df_qi = pd.read_csv(encoded_amazon_vectors_path, low_memory=False)
        df_qi["asin"] = df_qi["asin"].astype(str).str.strip().str.upper()


        vector_cols = df_qi.drop(columns=["asin"]).apply(pd.to_numeric, errors="coerce").astype(np.float32)
        df_qi_clean = pd.concat([df_qi["asin"], vector_cols], axis=1)

        # Rata-ratakan embedding jika 1 asin muncul lebih dari 1 kali
        df_qi_grouped = df_qi_clean.groupby("asin").mean()

        # Load mapping item_index → asin
        df_mapping = pd.read_csv(mapping_path)
        df_mapping["asin"] = df_mapping["asin"].astype(str).str.strip().str.upper()

        # Siapkan matrix kosong
        q_i = np.zeros((self.n_items, self.n_factors), dtype=np.float32)
        missing_count = 0

        for _, row in df_mapping.iterrows():
            item_idx = int(row["item_index"])
            asin = row["asin"]

            if asin in df_qi_grouped.index:
                vector = df_qi_grouped.loc[asin].to_numpy()
                if vector.shape[0] == self.n_factors:
                    q_i[item_idx] = vector
                else:
                    print(f"[WARNING] ASIN {asin} vektor tidak sesuai dimensi, di-skip.")
                    q_i[item_idx] = np.random.normal(scale=0.1, size=self.n_factors)
                    missing_count += 1
            else:
                print(f"[WARNING] ASIN {asin} tidak ditemukan di encoded vectors.")
                q_i[item_idx] = np.random.normal(scale=0.1, size=self.n_factors)
                missing_count += 1

        # Rescale q_i
        mean = np.mean(q_i)
        std = np.std(q_i)
        q_i_rescaled = (q_i - mean) / std * target_std

        print(f"[INFO] Finished loading item factors.")
        print(f"[INFO] Missing/invalid ASIN vectors: {missing_count}")
        print(f"[INFO] Shape of q_i: {q_i.shape}")
        print(f"[INFO] Rescaled q_i to target std: {target_std}")
        return q_i_rescaled


    def predict_all(self, df, q_i, output_path="prediction.csv"):
        """
        Simpan prediksi model untuk semua pasangan user-item yang tersedia di df.
        """
        result = []
        for _, row in df.iterrows():
            u = int(row["user_index"])
            i = int(row["item_index"])
            r_ui = float(row["rating"])
            # pred = np.dot(self.user_factors[u], q_i[i])
            pred = np.dot(self.user_factors[u], q_i[i])
            pred = np.clip(pred, 0.5, 5.0)
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