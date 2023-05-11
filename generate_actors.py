import pandas as pd
import numpy as np
import pickle as pkl

def main():
  # vectors: rating, numRatings, releaseYear, runtime, genres...
  actors = load_pickle('actors.pkl')

  a = {}
  for key, val in actors.items():
    if val > 15:
      a[key] = val
  actors = a

  actorsSize = len(actors)
  actorsConvert = load_pickle('actorsConvert.pkl')

  popular = pd.read_pickle('popular_movies.pkl')

  actorsVectors = []

  for index, row in popular.iterrows():
    vec = np.zeros(actorsSize)

    nconst = row['nconst']
    movie_names = []
    for n in nconst:
      if n in actorsConvert:
        movie_names.append(actorsConvert[n])
    # movie_names = [ actorsConvert[n] for n in nconst ]

    for ind, name in enumerate(actors):
      if name in movie_names:
        vec[ind] = 1

    actorsVectors.append(vec)

  with open('aVectors.pkl', 'wb') as f:
    pkl.dump(actorsVectors, f)

  return


def load_pickle(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)

if __name__ == '__main__':
  main()
