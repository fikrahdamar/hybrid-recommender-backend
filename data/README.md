## Isi Data

### Encoded

- tokenizer.pkl = token yang telah dibuat untuk processing text
- encoded_amazon_vectors.csv = data encoded dari amazon review digunakan sebagai q_i (item latent factors)
- encoded_amazon_vectors_new.csv = hasil dari encoded_amazon_vectors.csv yang telah diperbaiki

### Img

gambaran statistik dari hasil pelatihan data

### Mapping

- item_asin_mapping.csv = data yang telah di mapping antara movieId dan ASIN yang cocok

### Output

- test folder = percobaan hasil prediksi dari dataset
- item_factors.npy = item latent factors (q_i)
- user_factors.npy = user latent factors (p_u)
- pmf_training_data.csv = data p_u untuk di train

### Processed

- matched_amazon_reviews.jsonl = data amazon review telah difilter asin dan movieId
- matched_titles.csv = title asin dan movieId yang match hasil fuzzy matching
- ml_merged.csv = merged dari movielens_movies.dat dan movielens_ratings.dat
- used_asins.csv = asin yang telah difilter terpakai

### Raw_Data

- movielens_movies.dat
- movielens_ratings.dat
- Movies_and_TV_5.json
- meta_Movies_and_TV.json

### Sample

- sample_qi_vectors.csv = sample q_i untuk percobaan
