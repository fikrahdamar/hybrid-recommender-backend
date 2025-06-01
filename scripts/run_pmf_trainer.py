import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models import PMFTrainer


# Load dan preprocess data
full_df = pd.read_csv("./data/pmf_training_data.csv")
full_df["rating"] = (full_df["rating"] - 1) / 4.0  # Normalize rating ke range [0,1]

# Bikin mapping dari item_index lama ke index baru yang sequential
unique_items = full_df["item_index"].unique()
item_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(unique_items))}

# Update item_index di data pake mapping baru
full_df["item_index"] = full_df["item_index"].map(item_mapping)

# Split data jadi train dan validation
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

n_users = full_df["user_index"].nunique()
n_items = len(unique_items)  # Pake jumlah unique items yang bener

print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")

# Inisialisasi trainer dengan parameter
trainer = PMFTrainer(
    n_users=n_users,
    n_items=n_items,
    n_factors=512,  # Dimensi latent lebih besar
    lr=0.001,       # Learning rate kecil
    reg=0.001,      # Regularization lebih ringan
    epochs=100,     # Maksimum epochs
    patience=10,    # Early stopping patience
    min_lr=1e-6,    # Minimum learning rate
    decay_factor=0.98  # Learning rate decay
)

# Load item vectors dan align pake index
try:
    # Coba pake mapping file dulu kalo ada
    q_i = trainer.load_item_factors(
        "./data/encoded_amazon_vectors.csv",
        item_mapping_path="./data/item_mapping.csv"
    )
except FileNotFoundError:
    # Kalo ga ada mapping file, coba load langsung
    print("Item mapping file not found, attempting direct loading...")
    q_i = trainer.load_item_factors("./data/encoded_amazon_vectors.csv")

# Cek kalo ada item yang ga punya vector
missing_items = set(train_df["item_index"].unique()) - set(range(n_items))
if missing_items:
    print(f"Warning: Missing vectors for {len(missing_items)} items in training data")
    print("These items will use zero vectors for their latent factors")

# Training model
trainer.fit(train_df, q_i, val_df)
trainer.save_user_factors("./data/user_factors(tuned3).npy")
trainer.plot_losses("./data/img/loss_plot(tuned3).png")
