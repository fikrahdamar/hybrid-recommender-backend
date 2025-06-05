# scripts/run_pmf_trainer.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from models import PMF, load_amazon_vectors, load_movieid_to_asin, generate_id_mappings


def main():
    ratings_df = pd.read_csv('./data/ml_merged.csv')
    amazon_vectors, asin_to_index = load_amazon_vectors('./data/encoded_amazon_vectors.csv')
    movieid_to_asin = load_movieid_to_asin('./data/ml_amazon_matched.csv')

    # Split train/valid
    train_df, valid_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Buat mapping hanya dari train agar test tidak bocor
    user_mapping, item_mapping = generate_id_mappings(train_df)

    pmf = PMF(
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        num_factors=amazon_vectors.shape[1],
        learning_rate=0.01,
        reg=0.05,
        epochs=20
    )

    # Simpan RMSE untuk plot
    val_rmse_history = []

    for epoch in range(pmf.epochs):
        print(f"\n--- Epoch {epoch+1}/{pmf.epochs} ---")
        pmf.train(train_df, amazon_vectors, asin_to_index, movieid_to_asin, user_mapping, item_mapping)
        rmse = pmf.evaluate(valid_df, amazon_vectors, asin_to_index, movieid_to_asin, user_mapping, item_mapping)
        print(f"Validation RMSE: {rmse:.4f}")
        val_rmse_history.append(rmse)

    pmf.save_user_vectors('./data/output/user_vectors(newTuning).npy')

    # --- Plot RMSE ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, pmf.epochs + 1), val_rmse_history, marker='o', linestyle='-')
    plt.title('Validation RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig('./data/img/rmse_plot.png')
    print("Plot RMSE disimpan di output/rmse_plot.png")

    # --- Save Predictions to CSV ---
    preds, actuals, user_ids, movie_ids = [], [], [], []

    for _, row in valid_df.iterrows():
        uid = user_mapping.get(row['userId'])
        mid = item_mapping.get(row['movieId'])
        asin = movieid_to_asin.get(row['movieId'])

        if uid is None or mid is None or asin not in asin_to_index:
            continue

        q_i = amazon_vectors[asin_to_index[asin]]
        pred = np.dot(pmf.p_u[uid], q_i)

        preds.append(pred)
        actuals.append(row['rating'])
        user_ids.append(row['userId'])
        movie_ids.append(row['movieId'])

    df_pred = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'actual_rating': actuals,
        'predicted_rating': np.clip(preds, 0, 5)
    })

    df_pred.to_csv('./data/output/prediction.csv', index=False)
    print("Prediksi disimpan di output/prediction.csv")


if __name__ == '__main__':
    main()
