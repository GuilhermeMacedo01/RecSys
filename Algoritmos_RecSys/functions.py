def read_data(path):

    data = []
    with open(path) as f:
        for line in f:

            pieces = line.split()
            user_id = int(pieces[0])
            movie_id = int(pieces[1])
            rating = float(pieces[2])
            data.append((user_id, movie_id, rating))

    # user id | item id | rating | timestamp
    return data


def read_movies_name(path):
    data = {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            pieces = line.split('|')
            movie_id = int(pieces[0])
            title = pieces[1]
            data[movie_id] = title
        
    return data



def user_similarity(v1, v2):
    
    pearson = v1.corr(v2)
    
    return pearson