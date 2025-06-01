
import numpy as np
import pandas as pd


class PMF:
    def __init__(self, num_users, num_items, num_factors=100, learning_rate=0.005, reg=0.02, epochs=20):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs

        self.p_u = np.random.normal(0, 0.1, (num_users, num_factors))

    def train(self, ratings_df, q_i_amazon, asin_to_index, movieid_to_asin, user_mapping, item_mapping, valid_df=None):
        print("Mulai training PMF...")

        for epoch in range(self.epochs):
            total_loss = 0
            for _, row in ratings_df.iterrows():
                uid = user_mapping[row['userId']]
                mid = item_mapping[row['movieId']]
                asin = movieid_to_asin.get(row['movieId'])

                if asin is None or asin not in asin_to_index:
                    continue

                q_i = q_i_amazon[asin_to_index[asin]]
                r = row['rating']
                pred = np.dot(self.p_u[uid], q_i)
                err = r - pred

                self.p_u[uid] += self.lr * (err * q_i - self.reg * self.p_u[uid])
                total_loss += err ** 2 + self.reg * (np.linalg.norm(self.p_u[uid]) ** 2)

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {total_loss:.4f}")

            if valid_df is not None:
                rmse = self.evaluate(valid_df, q_i_amazon, asin_to_index, movieid_to_asin, user_mapping, item_mapping)
                print(f"  Validation RMSE: {rmse:.4f}")

    def evaluate(self, df, q_i_amazon, asin_to_index, movieid_to_asin, user_mapping, item_mapping):
        preds = []
        actuals = []
        for _, row in df.iterrows():
            uid = user_mapping.get(row['userId'])
            mid = item_mapping.get(row['movieId'])
            asin = movieid_to_asin.get(row['movieId'])

            if uid is None or mid is None or asin not in asin_to_index:
                continue

            q_i = q_i_amazon[asin_to_index[asin]]
            pred = np.dot(self.p_u[uid], q_i)
            preds.append(pred)
            actuals.append(row['rating'])

        preds = np.array(preds)
        actuals = np.array(actuals)
        return np.sqrt(np.mean((preds - actuals) ** 2))

    def save_user_vectors(self, path='output/user_vectors.npy'):
        np.save(path, self.p_u)
        print(f"User vectors disimpan di {path}")


def load_amazon_vectors(path):
    df = pd.read_csv(path)
    asin_col = df['asin']
    df = df.drop(columns=['asin'])
    vectors = df.values.astype(np.float32)
    asin_to_index = {asin: i for i, asin in enumerate(asin_col)}
    return vectors, asin_to_index


def load_movieid_to_asin(path):
    df = pd.read_csv(path)
    return dict(zip(df['movieId'], df['asin']))


def generate_id_mappings(ratings_df):
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()

    user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
    item_mapping = {mid: idx for idx, mid in enumerate(unique_items)}

    return user_mapping, item_mapping
