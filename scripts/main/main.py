from recommend import HybridRecommender,get_rated_item_indices
import os

user_factors_path = os.path.join("data", "output", "user_factors.npy")
item_factors_path = os.path.join("data", "output", "item_factors.npy")
item_mapping_path = os.path.join("data", "mapping", "item_asin_mapping_with_titles.csv")
training_data_path = os.path.join("data", "output", "pmf_training_data.csv")


recommender = HybridRecommender(user_factors_path, item_factors_path, item_mapping_path)
def get_recommendations_for_user(user_id, top_k=10):
    try:
       
        rated_items = get_rated_item_indices(training_data_path, user_id)
        watched_items = [
            {
                "item_index": item,
                "asin": recommender.index_to_asin.get(item, "UNKNOWN"),
                "title": recommender.index_to_title.get(item, "UNKNOWN")
            }
            for item in rated_items
        ]
    
        recommendations = recommender.recommend(user_id, top_k=top_k, exclude_items=rated_items)
        result = {
            "user_index": user_id,
            "watched_items": watched_items,
            "top_recommendations": recommendations
        }
        return result
    except Exception as e:
        return {"error": str(e)}

# Input user ID and number of top recommendations
user_id_input = int(input("Enter User ID: "))
top_k_input = int(input("Enter the number of top recommendations: "))
recommendation_result = get_recommendations_for_user(user_id_input, top_k_input)


import pprint
pprint.pprint(recommendation_result)