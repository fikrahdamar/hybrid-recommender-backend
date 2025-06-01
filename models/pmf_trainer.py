import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class PMFTrainer:
    def __init__(self, n_users, n_items, n_factors=256, lr=0.002, reg=0.01, epochs=70, 
                 patience=5, min_lr=1e-5, decay_factor=0.95):
        """
        Inisialisasi model PMF (Probabilistic Matrix Factorization)
        
        Parameters:
        -----------
        n_users : int
            Jumlah user yang ada di dataset
        n_items : int
            Jumlah item yang ada di dataset
        n_factors : int, default=256
            Jumlah dimensi untuk latent factors
        lr : float, default=0.002
            Learning rate untuk update weights 
        reg : float, default=0.01
            Regularization untuk mencegah overfitting 
        epochs : int, default=70
            Maksimum jumlah epoch training
        patience : int, default=5
            Berapa epoch harus nunggu kalau ga ada improvement (early stopping)
        min_lr : float, default=1e-5
            Learning rate minimum 
        decay_factor : float, default=0.95
            Faktor untuk menurunkan learning rate 
        """
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

    def load_item_factors(self, encoded_amazon_vectors_path, item_mapping_path=None):
        """
        Load dan align item vectors dari file CSV
        
        Parameters:
        -----------
        encoded_amazon_vectors_path : str
            Path ke file CSV yang isinya item vectors (dari LSTM encoder)
        item_mapping_path : str, optional
            Path ke file CSV yang isinya mapping antara asin dan item_index
            Kalau ga ada, asumsi asin = item_index
            
        Returns:
        --------
        q_i : numpy array
            Array yang isinya item vectors, udah dinormalize
            Shape: (n_items, n_factors)
        """
        # Load item vectors dengan dtype yang bener biar ga warning
        df_qi = pd.read_csv(encoded_amazon_vectors_path, dtype={str(i): float for i in range(256)})
        
        # Bikin array kosong untuk semua item
        q_i = np.zeros((self.n_items, self.n_factors))
        
        if item_mapping_path:
            # Kalau ada mapping file, pake itu untuk align indices
            mapping_df = pd.read_csv(item_mapping_path)
            asin_to_idx = dict(zip(mapping_df['asin'], mapping_df['item_index']))
            
            # Map setiap asin ke index yang bener
            for _, row in df_qi.iterrows():
                asin = row['asin']
                if asin in asin_to_idx:
                    idx = asin_to_idx[asin]
                    if idx < self.n_items:  # Pastiin index ga out of bounds
                        q_i[idx] = row.drop('asin').values
        else:
            # Kalau ga ada mapping, asumsi asin = item_index
            for _, row in df_qi.iterrows():
                try:
                    idx = int(row['asin'])
                    if idx < self.n_items:  # Cuma pake index yang valid
                        q_i[idx] = row.drop('asin').values
                except ValueError:
                    continue  # Skip kalo asin bukan angka
        
        # Normalize vectors biar ga ada masalah numerik
        q_i = q_i / (np.linalg.norm(q_i, axis=1, keepdims=True) + 1e-8)
        
        return q_i

    def evaluate_rmse(self, df, q_i):
        """
        Hitung RMSE untuk validation set
        
        Parameters:
        -----------
        df : pandas DataFrame
            Data yang mau di-evaluate (biasanya validation set)
        q_i : numpy array
            Item vectors yang udah di-load
            
        Returns:
        --------
        float
            RMSE score
        """
        errors = []
        for _, row in df.iterrows():
            u = int(row["user_index"])
            i = int(row["item_index"])
            r_ui = float(row["rating"])

            pred = np.dot(self.user_factors[u], q_i[i])
            errors.append((r_ui - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def fit(self, train_df, q_i, val_df=None):
        """
        Training model PMF
        
        Parameters:
        -----------
        train_df : pandas DataFrame
            Training data dengan kolom: user_index, item_index, rating
        q_i : numpy array
            Item vectors yang udah di-load
        val_df : pandas DataFrame, optional
            Validation data dengan format sama kayak train_df
        """
        no_improvement = 0
        
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

                # Update user factors pake gradient descent
                self.user_factors[u] += self.lr * (error * q_i_vec - self.reg * p_u)

                total_loss += error ** 2 + self.reg * (np.linalg.norm(p_u) ** 2)

            self.losses.append(total_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {total_loss:.4f}")

            if val_df is not None:
                val_rmse = self.evaluate_rmse(val_df, q_i)
                self.val_losses.append(val_rmse)
                print(f"Validation RMSE: {val_rmse:.4f}")
                
                # Cek early stopping
                if val_rmse < self.best_val_rmse:
                    self.best_val_rmse = val_rmse
                    self.best_epoch = epoch
                    self.best_user_factors = self.user_factors.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                # Learning rate decay kalo ga ada improvement
                if no_improvement > 0:
                    self.lr = max(self.min_lr, self.lr * self.decay_factor)
                    print(f"Learning rate decayed to: {self.lr:.6f}")
                
                # Stop kalo udah ga ada improvement
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
