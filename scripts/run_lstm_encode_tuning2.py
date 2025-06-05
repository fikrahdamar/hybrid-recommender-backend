import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import load_and_preprocess_reviews, prepare_tokenizer, texts_to_padded_sequences
from models import LSTMAutoencoder

# === Config ===
JSONL_PATH = './data/matched_amazon_reviews.jsonl'
BATCH_SIZE = 32
MAX_LEN = 300
VOCAB_SIZE = 20000
EMBED_DIM = 128
LSTM_UNITS = 256
EPOCHS = 5


texts, asin_list = load_and_preprocess_reviews(JSONL_PATH)
tokenizer = prepare_tokenizer(texts, num_words=VOCAB_SIZE)
sequences = texts_to_padded_sequences(tokenizer, texts, maxlen=MAX_LEN)
x = sequences.astype('int32')
y = sequences.astype('int32')
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


autoencoder = LSTMAutoencoder(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, lstm_units=LSTM_UNITS)
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train model ===
autoencoder.fit(dataset, epochs=EPOCHS)

# === Generate q_i ===
encoded_dataset = tf.data.Dataset.from_tensor_slices(sequences).batch(BATCH_SIZE)
all_embeddings = []
for batch in tqdm(encoded_dataset, desc="Generating q_i from encoder"):
    q_i = autoencoder.encode(batch)
    all_embeddings.append(q_i.numpy())

q_i_matrix = np.vstack(all_embeddings)

# === Save outputs ===
df_output = pd.DataFrame(q_i_matrix)
df_output['asin'] = asin_list
df_output.to_csv('./data/data_optimal_encoded/encoded_amazon_vectors.csv', index=False)

with open('./data/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

autoencoder.save_weights('./models/models_optimal/lstm_autoencoder_weights.h5')
print("✅ Encoding selesai dan disimpan.")

# === PCA Visualize ===
try:
    print("🔍 Visualizing 2D representation of q_i via PCA...")
    pca = PCA(n_components=2)
    q_i_2d = pca.fit_transform(q_i_matrix)

    plt.figure(figsize=(10, 6))
    plt.scatter(q_i_2d[:, 0], q_i_2d[:, 1], alpha=0.5, s=5)
    plt.title("2D PCA Visualization of Encoded Item Vectors (q_i)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.savefig('./data/img/q_i_pca_visualization_tuning2.png')
    plt.show()
except Exception as e:
    print(f"❌ PCA visualization failed: {e}")
