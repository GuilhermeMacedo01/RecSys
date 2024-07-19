import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functions import read_data, read_movies_name, user_similarity

MOVIE_RATINGS_PATH = './data/u.data'
MOVIE_NAMES_PATH = './data/u.item'


N_NEIGHBORS = 10
N_RECOMMENDATIONS = 5

def compute_similarities(user_id, ratings_matrix):
    ratings_user = ratings_matrix.loc[user_id, :]
    similarities = ratings_matrix.apply(lambda row: user_similarity(ratings_user, row), axis=1)
    similarities = similarities.to_frame(name='similarity')
    similarities = similarities.sort_values(by='similarity', ascending=False)
    similarities = similarities.drop(user_id)
    return similarities

def predict_rating(item_id, ratings, similarities, N=10):
    if item_id not in ratings.columns:
        print(f"Item ID {item_id} não encontrado.")
        return np.nan
    users_ratings = ratings.loc[:, item_id]
    most_similar_users_who_rated_item = similarities.loc[~users_ratings.isnull()]
    N_most_similar_users = most_similar_users_who_rated_item.head(N)
    ratings_for_item = ratings.loc[N_most_similar_users.index, item_id]
    return ratings_for_item.mean()

def recommend(user_id, ratings, movie_names, n_neighbors=10, n_recomm=5):
    ratings_wide = ratings.pivot(index='user', columns='movie', values='rating')
    all_items = ratings_wide.loc[user_id, :]
    unrated_items = all_items.loc[all_items.isnull()]
    unrated_items = unrated_items.index.to_series(name='item_ids').reset_index(drop=True)
    print(f'\nUser {user_id} tem {len(unrated_items)} itens não avaliados.')
    
    similarities = compute_similarities(user_id, ratings_wide)
    predictions = unrated_items.apply(lambda d: predict_rating(d, ratings_wide, similarities, N=n_neighbors))
    predictions = predictions.sort_values(ascending=False)
    recommends = predictions.head(n_recomm)
    recommends = recommends.to_frame(name='predicted_rating')
    recommends = recommends.rename_axis('movie_id')
    recommends = recommends.reset_index()
    recommends['name'] = recommends.movie_id.apply(lambda d: movie_names.get(d, 'Desconhecido'))
    return recommends

def predict(user_id, item_id, ratings):
    ratings_wide = ratings.pivot(index='user', columns='movie', values='rating')
    if item_id not in ratings_wide.columns:
        print(f"Item ID {item_id} não está presente no conjunto de dados.")
        return np.nan
    similarities = compute_similarities(user_id, ratings_wide)
    prediction = predict_rating(item_id, ratings_wide, similarities, N=N_NEIGHBORS)
    return prediction

def recommend_baseline(ratings, n_recomm=5):
    item_means = ratings.groupby('movie')['rating'].mean()
    top_items = item_means.sort_values(ascending=False).head(n_recomm)
    return top_items

def evaluate_model(ratings, n_splits=5):
    kf = KFold(n_splits=n_splits)
    user_based_rmses, user_based_maes = [], []
    baseline_rmses, baseline_maes = [], []
    
    start_time = time.time() 
    for train_index, test_index in kf.split(ratings):
        train_ratings, test_ratings = ratings.iloc[train_index], ratings.iloc[test_index]
        
        user_based_predictions, true_ratings = [], []
        for _, row in test_ratings.iterrows():
            user_id, item_id, true_rating = row['user'], row['movie'], row['rating']
            predicted_rating = predict(user_id, item_id, train_ratings)
            if not np.isnan(predicted_rating):
                user_based_predictions.append(predicted_rating)
                true_ratings.append(true_rating)
        
        if user_based_predictions:
            user_based_rmse = mean_squared_error(true_ratings, user_based_predictions, squared=False)
            user_based_mae = mean_absolute_error(true_ratings, user_based_predictions)
            user_based_rmses.append(user_based_rmse)
            user_based_maes.append(user_based_mae)
        
        baseline_predictions, true_ratings = [], []
        baseline_top_items = recommend_baseline(train_ratings, n_recomm=10)
        for _, row in test_ratings.iterrows():
            item_id, true_rating = row['movie'], row['rating']
            if item_id in baseline_top_items.index:
                predicted_rating = baseline_top_items.get(item_id, np.nan)
                if not np.isnan(predicted_rating):
                    baseline_predictions.append(predicted_rating)
                    true_ratings.append(true_rating)
        
        if baseline_predictions:
            baseline_rmse = mean_squared_error(true_ratings, baseline_predictions, squared=False)
            baseline_mae = mean_absolute_error(true_ratings, baseline_predictions)
            baseline_rmses.append(baseline_rmse)
            baseline_maes.append(baseline_mae)
    
    end_time = time.time()
    print(f"Tempo para avaliar o modelo: {end_time - start_time:.2f} segundos")
    
    return (np.mean(user_based_rmses), np.mean(user_based_maes)), (np.mean(baseline_rmses), np.mean(baseline_maes))

if __name__ == '__main__':
    ratings = read_data(MOVIE_RATINGS_PATH)
    ratings = pd.DataFrame(data=ratings, columns=['user', 'movie', 'rating'])
    ratings = ratings.astype(int)
    movie_names = read_movies_name(MOVIE_NAMES_PATH)

    (user_based_rmse, user_based_mae), (baseline_rmse, baseline_mae) = evaluate_model(ratings)
    print(f'User-Based RMSE: {user_based_rmse}')
    print(f'User-Based MAE: {user_based_mae}')
    print(f'Baseline RMSE: {baseline_rmse}')
    print(f'Baseline MAE: {baseline_mae}')
