import pandas as pd
import numpy as np
import pickle as pkl

def main():
  # vectors: rating, numRatings, releaseYear, runtime, genres...
  directors = load_pickle('directors.pkl')

  d = {}
  for key, val in directors.items():
    if val > 5:
      d[key] = val
  directors = d

  directorsSize = len(directors)
  directorsConvert = load_pickle('directorsConvert.pkl')

  popular = pd.read_pickle('popular_movies.pkl')

  directorsVectors = []

  for index, row in popular.iterrows():
    vec = np.zeros(directorsSize)

    nconst = row['nconst']
    movie_names = []
    for n in nconst:
      if n in directorsConvert:
        movie_names.append(directorsConvert[n])
    # movie_names = [ actorsConvert[n] for n in nconst ]

    for ind, name in enumerate(directors):
      if name in movie_names:
        vec[ind] = 1

    directorsVectors.append(vec)

  with open('dVectors.pkl', 'wb') as f:
    pkl.dump(directorsVectors, f)

  return


def load_pickle(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)

if __name__ == '__main__':
  main()
