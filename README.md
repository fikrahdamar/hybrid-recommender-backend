# Hybrid Recommender

Hybrid Recommender adalah sistem untuk merekomendasi film untuk tiap user menggunakan pendekatan Hybrid yaitu Collaborative filtering yang dimana menggunakan metode PMF untuk mendapatkan user latent factor dan juga Content-Based Filtering yang menggunakan LSTM (Long Short Term Memory) yang dimana untuk menghasilkan item latent factor yang optimal dan dikolaborasikan

untuk melihat bagian front-end dari Hybrid Recommender ini bisa klik link dibawah :
[Hybrid-Recommender-FrontEnd](https://github.com/iqbalraihanfr/hybrid-recommender-frontend)

## Setup

jika anda ingin mencoba dari awal
Pastikan telah memiliki dataset berikut ini pada komputer :

- [movielens dataset](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset)
- [amazon review dataset](https://nijianmo.github.io/amazon/index.html#subsets)

## Process Data

tahap percobaan dari awal yaitu

- lakukan penggabungan data pada movielens_movies and movielens_ratings
- lakukan pencocokan title pada movielens_movies dan meta_Movies_and_TV.json
- lakukan proses lstm encode untuk menghasilkan encode item latent factors (q_i)
- simpan hasil dari q_i
- lakukan mapping ada movieId, dan asin untuk mendapatkan item_index sesuai asin
- lakukan pelatihan data user item factors (p_u)
- gabungkan item lateng factors dan user item factors untuk mendapatkan rekomendasi dari top-10

## Library

library yang digunakan sebagai berikut

- pandas `pip install pandas`
- matplotlib `pip install matplotlib`
- numpy `pip install numpy`
- tensorflow `pip install tensorflow==2.10.0`
- skicit-learn `pip install skicit-learn`
- tqdm `pip install tqdm`
-

## Usage

untuk langsung menggunakan api dari hybrid-recommender
lakukan hal dibawah ini :

- install fastapi:
  `pip install "fastapi[standard]"`

- install uvicorn:
  `$ pip install 'uvicorn[standard]'`

- jalankan project:
  `uvicorn api.app:app --reload`
