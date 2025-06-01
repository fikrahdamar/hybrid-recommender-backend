import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class PMFTrainer:
    def __init__(self, n_users, n_items, n_factors=256, lr=0.01, reg=0.001, epochs=40, 
                 patience=5, min_lr=1e-5, decay_factor=0.9):
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.initial_lr = lr
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.patience = patience
        self.min_lr = min_lr
        self.decay_factor = decay_factor

        # Inisialisasi user factors dengan random values
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, n_factors))  # p_u
        self.losses = []  # Untuk nyimpen history loss
        self.val_losses = []  # Untuk nyimpen history validation RMSE
        self.best_val_rmse = float('inf')  # Best validation RMSE so far
        self.best_epoch = 0  # Epoch terbaik
        self.best_user_factors = None  # User factors terbaik


    def denormalize_rating(self, r):
        return r * 4.0 + 1.0  
    
    
    def load_item_factors(self, path, item_mapping, asin_map_path):

        # Load encoded vectors dan asin-nya
        df = pd.read_csv(path)

        # ASIN ada di kolom terakhir
        asin_ids = df.iloc[:, -1].values
        asin_vectors = df.iloc[:, :-1].values  # Semua kolom kecuali terakhir

        # Peta ASIN ke vektor
        asin_to_vector = {asin: vec for asin, vec in zip(asin_ids, asin_vectors)}

        # Load mapping item_index (original MovieLens) ke ASIN
        asin_map_df = pd.read_csv(asin_map_path)
        asin_map = dict(zip(asin_map_df["item_index"], asin_map_df["asin"]))

        # Siapkan matriks q_i
        n_items = len(item_mapping)
        vector_dim = asin_vectors.shape[1]
        q_i = np.zeros((n_items, vector_dim))

        for original_idx, mapped_idx in item_mapping.items():
            asin = asin_map.get(original_idx)
            if asin in asin_to_vector:
                q_i[mapped_idx] = asin_to_vector[asin]

        return q_i

    def evaluate_rmse(self, df, q_i, save_path=None):
        preds = []
        trues = []

        for _, row in df.iterrows():
            u = int(row["user_index"])
            i = int(row["item_index"])
            r_ui = float(row["rating"])

            pred = np.dot(self.user_factors[u], q_i[i])
            preds.append(pred)
            trues.append(r_ui)

        rmse = np.sqrt(np.mean((np.array(trues) - np.array(preds)) ** 2))

        if save_path:
            pd.DataFrame({"true_rating": trues, "predicted_rating": preds}).to_csv(save_path, index=False)
            print(f"Saved predictions to {save_path}")

        return rmse

    def fit(self, train_df, q_i, val_df=None):
        no_improvement = 0
        
        # Optional: normalize q_i vectors to unit length
        q_i = q_i / (np.linalg.norm(q_i, axis=1, keepdims=True) + 1e-8)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch+1}"):
                u = int(row["user_index"])
                i = int(row["item_index"])
                r_ui = float(row["rating"])

                p_u = self.user_factors[u]
                q_i_vec = q_i[i]

                pred = np.dot(p_u, q_i_vec)  # Tanpa clip/denormal
                error = r_ui - pred

                # Update user factors
                self.user_factors[u] += self.lr * (error * q_i_vec - self.reg * p_u)

                total_loss += error ** 2 + self.reg * (np.linalg.norm(p_u) ** 2)
                
            avg_loss = total_loss / len(train_df)
            self.losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f}")

            if val_df is not None:
                val_rmse = self.evaluate_rmse(val_df, q_i)
                self.val_losses.append(val_rmse)
                print(f"Validation RMSE: {val_rmse:.4f}")
                
                if val_rmse < self.best_val_rmse:
                    self.best_val_rmse = val_rmse
                    self.best_epoch = epoch
                    self.best_user_factors = self.user_factors.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                    self.lr = max(self.min_lr, self.lr * self.decay_factor)
                    print(f"Learning rate decayed to: {self.lr:.6f}")
                    
                if no_improvement >= self.patience:
                    print(f"\nEarly stopping triggered! No improvement for {self.patience} epochs.")
                    print(f"Best validation RMSE: {self.best_val_rmse:.4f} at epoch {self.best_epoch + 1}")
                    self.user_factors = self.best_user_factors
                    break

    def save_user_factors(self, path="user_factors.npy"):
        """
        Simpan user factors ke file
        
        Parameters:
        -----------
        path : str
            Path untuk nyimpen file .npy
        """
        np.save(path, self.user_factors)
        print(f"User factors saved to {path}")

    def plot_losses(self, path=None):
        """
        Plot training loss dan validation RMSE
        
        Parameters:
        -----------
        path : str, optional
            Path untuk nyimpen plot (kalau ga ada, plot langsung ditampilin)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.losses) + 1), self.losses, marker='o', label='Train Loss')
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, marker='x', label='Validation RMSE')
            plt.axvline(x=self.best_epoch + 1, color='r', linestyle='--', 
                       label=f'Best Epoch ({self.best_epoch + 1})')
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
