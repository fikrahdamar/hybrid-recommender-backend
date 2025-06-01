import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models import PMFTrainer

# === Load dan Preprocessing ===
full_df = pd.read_csv("./data/pmf_training_data.csv")
full_df["rating"] = (full_df["rating"] - 1) / 4.0  # Normalisasi ke [0, 1]

# Simpan item_index asli sebelum mapping
full_df["original_item_index"] = full_df["item_index"]

# Mapping item_index ke index sequential
unique_items = full_df["original_item_index"].unique()
item_mapping = {old: new for new, old in enumerate(sorted(unique_items))}
full_df["item_index"] = full_df["original_item_index"].map(item_mapping)

# Train-val split
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

n_users = full_df["user_index"].nunique()
n_items = len(unique_items)

# === Inisialisasi PMFTrainer ===
trainer = PMFTrainer(
    n_users=n_users,
    n_items=n_items,
    n_factors=256,
    lr=0.01,
    reg=0.001,
    epochs=30,
    patience=10,
    min_lr=1e-6,
    decay_factor=0.9
)

# === Load encoded item vectors (q_i) dengan mapping ===
try:
    q_i = trainer.load_item_factors(
        path="./data/encoded_amazon_vectors.csv",
        item_mapping=item_mapping,
        asin_map_path="./data/item_asin_mapping.csv"
    )
except FileNotFoundError:
    print("❌ File encoded_amazon_vectors.csv atau mapping tidak ditemukan.")
    exit()

# Validasi item index
missing_items = set(train_df["item_index"].unique()) - set(range(len(q_i)))
if missing_items:
    print(f"⚠️ Warning: {len(missing_items)} items tidak punya vektor, akan pakai vektor nol.")

# Opsional: cek jumlah vektor nol
zero_vector_count = np.sum(np.linalg.norm(q_i, axis=1) == 0)
print(f"ℹ️ Total item dengan vektor nol: {zero_vector_count}")

# === Training dan Evaluasi ===
trainer.fit(train_df, q_i, val_df)
trainer.evaluate_rmse(val_df, q_i, save_path="./data/predictions_after_train.csv")
trainer.save_user_factors("./data/user_factors(tuned4).npy")
trainer.plot_losses("./data/img/loss_plot(tuned4).png")
