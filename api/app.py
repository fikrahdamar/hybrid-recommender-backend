from fastapi import FastAPI, HTTPException, Query
from recommend import HybridRecommender, get_rated_item_indices
from typing import List
import os

app = FastAPI()

# Load recommender saat server startup
user_factors_path = os.path.join("data", "output", "user_factors.npy")
item_factors_path = os.path.join("data", "output", "item_factors.npy")
item_mapping_path = os.path.join("data", "mapping", "item_asin_mapping_with_titles.csv")
# item_mapping_path = os.path.join("data", "mapping", "item_asin_mapping.csv")
training_data_path = os.path.join("data", "output", "pmf_training_data.csv")

recommender = HybridRecommender(user_factors_path, item_factors_path, item_mapping_path)

@app.get("/recommend")
def get_recommendation(id: int = Query(..., description="User index"), top_k: int = 10):
    try:
        
        rated_items = get_rated_item_indices(training_data_path, id)
        watched_items = [
            {
                "item_index": item,
                "asin": recommender.index_to_asin.get(item, "UNKNOWN"),
                "title": recommender.index_to_title.get(item, "UNKNOWN") 
            }
            for item in rated_items
        ]
        
        recommendations = recommender.recommend(id, top_k=top_k, exclude_items=rated_items)

        return {
            "user_index": id,
            "watched_items": watched_items,
            "top_recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
