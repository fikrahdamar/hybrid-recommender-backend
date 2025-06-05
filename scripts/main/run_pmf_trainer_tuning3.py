import pandas as pd
from sklearn.model_selection import train_test_split
from models import PMFTrainer3
import numpy as np


full_df = pd.read_csv("./data/pmf_training_data.csv", low_memory=False)
n_items = full_df["item_index"].nunique()
print(f"Jumlah item unik dari pmf_training_data.csv: {n_items}")

full_df["rating"] = (full_df["rating"] - 1) / 4.0
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

n_users = full_df["user_index"].nunique()
        
# Inisialisasi trainer
trainer = PMFTrainer3(n_users=n_users, n_items=n_items, n_factors=256, lr=0.001, reg=0.2, epochs=30)
q_i = trainer.load_item_factors(
    encoded_amazon_vectors_path="./data/data_optimal_encoded/encoded_amazon_vectors_new.csv",
    mapping_path="./data/item_asin_mapping.csv",
    target_std=0.1
)


print("Mean q_i:", np.mean(q_i))
print("Std q_i:", np.std(q_i))
print("Min q_i:", np.min(q_i))
print("Max q_i:", np.max(q_i))
# Latih model
trainer.fit(train_df, q_i, val_df)
print("Contoh vektor user pertama:", trainer.user_factors[0][:10])
np.save("./data/final_data/q_i(newTuned_targetStd01_reg02).npy", q_i)
trainer.save_user_factors("./data/final_data/user_factors(newTuned_targetStd01_reg02).npy")
trainer.plot_losses("./data/img/loss_plot(newTuned_targetStd01_reg02).png")
trainer.predict_all(val_df, q_i, output_path="./data/output/prediction2(newTuned_targetStd01_reg02).csv")