# RecSys

# Sistema de Recomendação Baseado em Usuário

Este repositório contém uma implementação de um sistema de recomendação baseado em memória e baseada em modelo utilizando o conjunto de dados MovieLens 100k. O projeto inclui a avaliação do modelo de recomendação com métricas como MAE (Mean Absolute Error), RMSE e também fornece um benchmark de tempo para a avaliação do modelo.

## Estrutura do Projeto

  - **Funções:**
    - `compute_similarities(user_id, ratings_matrix)`: Calcula a similaridade entre usuários.
    - `predict_rating(item_id, ratings, similarities, N=10)`: Prever a classificação de um item com base na similaridade dos usuários.
    - `recommend(user_id, ratings, movie_names, n_neighbors=10, n_recomm=5)`: Gera recomendações para um usuário específico.
    - `predict(user_id, item_id, ratings)`: Faz uma previsão de classificação para um usuário e item específicos.
    - `recommend_baseline(ratings, n_recomm=5)`: Método de recomendação baseado na média das avaliações dos itens.
    - `evaluate_model(ratings, n_splits=5)`: Avalia o modelo utilizando validação cruzada e calcula o MAE. Também mede o tempo de avaliação do modelo.

  - `u.data`: Dados de classificações dos filmes.
  - `u.item`: Dados dos filmes, incluindo nomes e IDs.
  - `functions.py`: Funções para ler dados e calcular a similaridade entre usuários.

## Requisitos

- Python 3.x
- Bibliotecas Python:
  - pandas
  - numpy
  - scikit-learn


