import pandas as pd
import numpy as np
import pickle as pkl

def main():
  # vectors: rating, numRatings, releaseYear, runtime, genres...
  runtimes = load_pickle('runtimes.pkl')
  years = load_pickle('years.pkl')
  genres = load_pickle('genres.pkl')
  genres = list(genres)
  numRatings = load_pickle('numRatings.pkl')
  popular_movies = pd.read_pickle('popular_movies.pkl')

  scale = lambda mi,ma,val: (val-mi)/(ma-mi)

  vectors = []

  for index, row in popular_movies.iterrows():
    vec = np.zeros(1 + 1 + 1 + 1 + len(genres)) # rating, numRatings, runtimes, years, genres
    vec[0] = scale(1, 10, float(row['averageRating']))
    vec[1] = scale(min(numRatings), max(numRatings), int(row['numVotes']))
    try:
      vec[2] = scale(min(runtimes), max(runtimes), int(row['runtimeMinutes']))
    except:
      vec[2] = 0

    vec[3] = scale(min(years),max(years), int(row['startYear']))

    gen = row['genres'].lower().split(',')
    for ind, g in enumerate(genres):
      if g in gen:
        vec[4 + ind] = 1

    vectors.append((row['primaryTitle'], vec))

  with open('vectors.pkl', 'wb') as f:
    pkl.dump(vectors,f)

  return


def load_pickle(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)

if __name__ == '__main__':
  main()