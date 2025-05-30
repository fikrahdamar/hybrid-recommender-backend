import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import preprocess_title

def load_and_preprocess_reviews(jsonl_path, max_samples=None):
    texts = []
    asin_list = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            summary = item.get('summary', '')
            review = item.get('reviewText', '')
            combined = preprocess_title(summary + ' ' + review)
            texts.append(combined)
            asin_list.append(item['asin'])
            if max_samples and len(texts) >= max_samples:
                break
    return texts, asin_list

def prepare_tokenizer(texts, num_words=20000):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_padded_sequences(tokenizer, texts, maxlen=300):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    return padded

def build_tf_dataset(sequences, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(sequences)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(sequences))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds