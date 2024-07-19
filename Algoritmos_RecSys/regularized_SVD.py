import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

def read_data(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['user', 'movie', 'rating', 'timestamp']).drop('timestamp', axis=1)

def read_movies_name(file_path):
    movie_data = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1', usecols=[0, 1])
    return movie_data.set_index(0)[1].to_dict()

MOVIE_RATINGS_PATH = './data/u.data'
MOVIE_NAMES_PATH = './data/u.item'

ratings = read_data(MOVIE_RATINGS_PATH)
ratings = ratings.astype(int)
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

n_users = ratings['user'].nunique()
n_items = ratings['movie'].nunique()
utility_matrix = np.zeros((n_users, n_items))

for line in train_data.itertuples():
    utility_matrix[line[1]-1, line[2]-1] = line[3]

def regularized_svd(R, k=20, lambda_reg=0.1, learning_rate=0.01, epochs=20):
    m, n = R.shape
    U = np.random.normal(scale=1./k, size=(m, k))
    V = np.random.normal(scale=1./k, size=(n, k))
    
    for epoch in range(epochs):
        for i in range(m):
            for j in range(n):
                if R[i, j] > 0:
                    error = R[i, j] - np.dot(U[i, :], V[j, :].T)
                    for l in range(k):
                        U[i, l] += learning_rate * (2 * error * V[j, l] - lambda_reg * U[i, l])
                        V[j, l] += learning_rate * (2 * error * U[i, l] - lambda_reg * V[j, l])
    return U, V

def evaluate_svd(utility_matrix, k, lambda_reg, learning_rate, epochs):
    U, V = regularized_svd(utility_matrix, k=k, lambda_reg=lambda_reg, learning_rate=learning_rate, epochs=epochs)
    predicted_ratings = np.dot(U, V.T)
    test_ratings = utility_matrix[test_data['user']-1, test_data['movie']-1]
    mask = (test_ratings > 0)
    rmse = np.sqrt(mean_squared_error(test_ratings[mask], predicted_ratings[test_data['user']-1, test_data['movie']-1][mask]))
    return rmse

param_grid = {
    'k': [10, 20],
    'lambda_reg': [0.01, 0.1],
    'learning_rate': [0.01, 0.1],
    'epochs': [10, 20]
}

best_rmse = float('inf')
best_params = None

start_time = time.time()

for k in param_grid['k']:
    for lambda_reg in param_grid['lambda_reg']:
        for learning_rate in param_grid['learning_rate']:
            for epochs in param_grid['epochs']:
                rmse = evaluate_svd(utility_matrix, k, lambda_reg, learning_rate, epochs)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (k, lambda_reg, learning_rate, epochs)

end_time = time.time()
print(f'regularized_SVD RMSE: {best_rmse}')
print(f'Melhores Parametros: k={best_params[0]}, lambda_reg={best_params[1]}, learning_rate={best_params[2]}, epochs={best_params[3]}')
print(f'Tempo para avaliar o modelo: {end_time - start_time} segundos')
