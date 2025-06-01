import pandas as pd
from sklearn.model_selection import train_test_split
from models import PMFTrainer3


full_df = pd.read_csv("./data/pmf_training_data.csv")
n_items = full_df["item_index"].nunique()
print(f"Jumlah item unik dari pmf_training_data.csv: {n_items}")

full_df["rating"] = (full_df["rating"] - 1) / 4.0
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

n_users = full_df["user_index"].nunique()
n_items = full_df["item_index"].nunique()

# Inisialisasi trainer
trainer = PMFTrainer3(n_users=n_users, n_items=n_items, n_factors=256, lr=0.0005, reg=0.01, epochs=30)
q_i, valid_items = trainer.load_item_factors(
    "./data/encoded_amazon_vectors.csv",
    "./data/item_asin_mapping.csv"
)

train_df_clean = train_df[train_df["item_index"].isin(valid_items)].copy()
val_df_clean = val_df[val_df["item_index"].isin(valid_items)].copy()

# Latih model
print("Jumlah data train sebelum filter:", len(train_df))
print("Jumlah data train setelah filter:", len(train_df_clean))
print("Jumlah data validasi setelah filter:", len(val_df_clean))
# trainer.fit(train_df_clean, q_i, val_df_clean)
# trainer.save_user_factors("./data/user_factors(newTuned).npy")
# trainer.plot_losses("./data/img/loss_plot(newTuned).png")
# trainer.predict_all(val_df, q_i, output_path="./data/output/prediction.csv")