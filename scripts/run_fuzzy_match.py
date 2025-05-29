import json
import pandas as pd
from preprocessing import get_best_matches
import matplotlib.pyplot as plt


# load data ml_merged buat sourcenya
ml_df = pd.read_csv('./data/ml_merged.csv')
source_titles= ml_df['title'].unique().tolist()

# load metadata amazon buat targetnya
with open('./data/meta_Movies_and_TV.json', 'r') as f:
    amazon_data = [json.loads(line) for line in f]
    
amazon_titles = {}
for item in amazon_data:
    if 'title' in item and 'asin' in item:
        title= item['title'].strip()
        if title:
            amazon_titles[title] = item['asin']

target_titles = list(amazon_titles.keys())
print("Fuzzy Matching Started...")

matches = get_best_matches(source_titles, target_titles, threshold=90)

results = []
for source, (matched_title, score) in matches.items():
    asin = amazon_titles[matched_title]
    matched_row = ml_df[ml_df['title'] == source]
    if not matched_row.empty:
        movie_id = matched_row.iloc[0]['movieId']
    else:
        movie_id = None
    results.append({
        'movieId': movie_id,
        'movielens_title' : source,
        'amazon_title' : matched_title,
        'score' : score,
        'asin' : asin
    })

matched_df = pd.DataFrame(results)
matched_df.to_csv('./data/matched_titles.csv', index=False)
print(f"Saved {len(matched_df)} matched titles.")

plt.figure(figsize=(8, 5))
plt.hist(matched_df['score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribusi Skor Fuzzy Matching')
plt.xlabel('Fuzzy Score')
plt.ylabel('Jumlah Judul')
plt.grid(True)
plt.tight_layout()
plt.savefig('./data/img/fuzzy_score_distribution.png')
plt.show()

