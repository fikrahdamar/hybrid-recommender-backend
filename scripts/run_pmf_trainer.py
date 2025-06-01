import pandas as pd
from sklearn.model_selection import train_test_split
from models import PMFTrainer


full_df = pd.read_csv("./data/pmf_training_data.csv")
full_df["rating"] = (full_df["rating"] - 1) / 4.0
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

n_users = full_df["user_index"].nunique()
n_items = full_df["item_index"].nunique()

# Inisialisasi trainer
trainer = PMFTrainer(n_users=n_users, n_items=n_items, n_factors=256, lr=0.001, reg=0.05, epochs=30)
q_i = trainer.load_item_factors("./data/encoded_amazon_vectors.csv")
trainer.evaluate_rmse(val_df, q_i, save_path="./data/predictions.csv")

# Latih model
trainer.fit(train_df, q_i, val_df)
trainer.evaluate_rmse(val_df, q_i, save_path="./data/predictions_after_train.csv")
trainer.save_user_factors("./data/user_factors(tuned1).npy")
trainer.plot_losses("./data/img/loss_plot(tuned1).png")
