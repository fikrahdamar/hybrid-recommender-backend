import pandas as pd

df_qi_raw = pd.read_csv("./data/encoded_amazon_vectors.csv")
df_qi_raw["asin"] = df_qi_raw["asin"].astype(str).str.strip().str.upper()

df_mapping = pd.read_csv("./data/item_asin_mapping.csv")
df_mapping["asin"] = df_mapping["asin"].astype(str).str.strip().str.upper()

# Cek seberapa banyak ASIN dari mapping yang cocok di vektor Amazon
matched_asin = df_mapping["asin"].isin(df_qi_raw["asin"])
print("Jumlah ASIN yang match:", matched_asin.sum(), "dari total", len(df_mapping))