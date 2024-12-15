import pandas as pd
import numpy as np

movies = pd.read_csv('https://liangfgithub.github.io/MovieData/movies.dat', sep='::', engine='python', header=None, encoding='latin-1')
movies.columns = ['MovieID', 'Title', 'Genres']

genres = list(
    sorted(set([genre for genres in movies.Genres.unique() for genre in genres.split("|")]))
)
def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(newuser):
    popularity = pd.read_csv('popularity.csv')
    popularity_ranking = popularity[['MovieID', 'weighted_rating']].sort_values('weighted_rating', ascending=False)
    
    similarity_matrix = pd.read_csv('similarity_matrix.csv',index_col=0)
    
    sim = similarity_matrix.fillna(0).values
    rated_movies = np.array([i for i in newuser.index if not np.isnan(newuser[i])])
    rows_matrix = np.array([similarity_matrix.index] * len(similarity_matrix.columns)).T
    columns_matrix = np.array([similarity_matrix.columns] * len(similarity_matrix.index))
    
    predicted_rating = {}
    for i in range(len(sim)):
        denom = 0
        num = 0
        for j in range(len(sim)):
            if sim[i,j]!=0 and columns_matrix[i,j] in rated_movies:
                denom+=sim[i,j]
                num+=(sim[i,j]*newuser[columns_matrix[i,j]])
        if denom>0 and rows_matrix[i,j] not in predicted_rating and rows_matrix[i,j] not in rated_movies:
            predicted_rating[rows_matrix[i,j]]= num/denom
    
    predicted_rating = dict(sorted(predicted_rating.items(), key=lambda item: item[1], reverse=True))
    
    rec_movies = list(predicted_rating.keys())[:10]
    rec_ratings = list(predicted_rating.values())[:10]
    
    rated_ids = np.array([int(i[1:]) for i in newuser.index if not np.isnan(newuser[i])])
    if len(rec_movies) < 10:
        remaining = 10 -len(rec_movies)
        available = popularity_ranking.loc[~popularity_ranking['MovieID'].isin(rated_ids)]
        top = available.head(remaining)
        rec_movies.extend(('m'+top['MovieID'].astype(str)).values)
        rec_ratings.extend(top['weighted_rating'].values)

    return pd.DataFrame({
        'MovieID': [int(movie_id[1:]) for movie_id in rec_movies],
        'Rating': rec_ratings,
        'Title': [movies[movies['MovieID'] == int(movie_id[1:])]['Title'].values[0] for movie_id in rec_movies]

    })