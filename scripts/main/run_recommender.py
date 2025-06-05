from recommend import HybridRecommender
import pandas as pd

def get_rated_item_indices(csv_path, user_index):
    df = pd.read_csv(csv_path)
    rated = df[df['user_index'] == user_index]['item_index'].tolist()
    return rated


recommender = HybridRecommender(
    user_factors_path="./data/output/user_factors.npy",
    item_factors_path="data/output/item_factors.npy",
    item_mapping_path="./data/mapping/item_asin_mapping.csv"
)


user_index = 0
rated_item_indices = get_rated_item_indices("./data/output/pmf_training_data.csv", user_index)
recommendations = recommender.recommend(user_index, top_k=10)

recommendations = [rec for rec in recommendations if rec['item_index'] not in rated_item_indices]

for rec in recommendations:
    print(f"Rank {rec['rank']}: ASIN {rec['asin']} (score: {rec['score']:.4f})")
