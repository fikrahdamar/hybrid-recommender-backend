import json
import pandas as pd
import os

def load_jsonl(file_path):
    """Load JSON Lines file (satu objek JSON per baris)"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    """Simpan list of dict sebagai JSON Lines"""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_csv(file_path):
    return pd.read_csv(file_path)

def save_csv(df, file_path):
    df.to_csv(file_path, index=False)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

