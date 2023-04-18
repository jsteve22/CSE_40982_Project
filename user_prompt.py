import numpy as np
import pandas as pd
import pickle as pkl

cosine = lambda x,y: np.dot(x,y) / ( np.linalg.norm(x)*np.linalg.norm(y)) if (np.linalg.norm(x)*np.linalg.norm(y)) != 0 else 0
bound = lambda array: np.minimum( np.ones(len(array)), np.maximum( np.zeros(len(array)), array) )

def main():
  # get vectors for all movies
  moviesVecs = pkl_load('vectors.pkl')
  actors     = pkl_load('actorsVectors.pkl')
  directors  = pkl_load('directorsVectors.pkl')

  names  = [n for n, _ in moviesVecs]
  movies = [v for _, v in moviesVecs]

  # generate base vector for user preference
  userMovies    = np.ones( len(movies[0]), dtype=float ) / 2
  userActors    = np.ones( len(actors[0]), dtype=float ) / 2 
  userDirectors = np.ones( len(directors[0]), dtype=float ) / 2

  # read all of the most popular movies with 
  popular = pd.read_pickle('popular_movies.pkl')
  popular = popular.reset_index(drop=True)
  # print(popular)
  # print(movies[0])

  top_100 = popular[popular['numVotes'] > 50_000].sort_values(by=['numVotes', 'averageRating', 'startYear'], ascending=False).iloc[:100]

  userSaw = []

  for _ in range(5):
    randomMovieIndex = np.random.randint(0,100)
    randomMovie = top_100.iloc[randomMovieIndex]
    randomMovieName = randomMovie['primaryTitle']
    userSaw.append(randomMovieName)
    index = names.index( randomMovieName )
    print(f'What do you think about {randomMovieName}?')
    print(f'\t1 if you like it')
    print(f'\t2 if you do not like it')
    print(f'\t3 if you do not know it')
    userInput = int(input())
    if (userInput == 1):
      userMovies    = userMovies + (0.3 * movies[index] )
      userActors    = userActors + (0.3 * actors[index] )
      userDirectors = userDirectors + (0.3 * directors[index] )
    elif (userInput == 2):
      userMovies    = userMovies - (0.3 * movies[index] )
      userActors    = userActors - (0.3 * actors[index] )
      userDirectors = userDirectors - (0.3 * directors[index] )
      pass
    elif (userInput == 3):
      pass
  
  userMovies    = bound(userMovies)
  userActors    = bound(userActors)
  userDirectors = bound(userDirectors)

  print()
  print(userMovies)
  print(userActors)
  print(userDirectors)
  print()
  print()

  user = (userMovies, userActors, userDirectors)
  data = (names, movies, actors, directors)

  for movie in topSimilar(user,data):
    if movie not in userSaw:
      print(movie)

  return 

def pkl_load(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)

def topSimilar(user, data, num=10):
  names, movies, actors, directors = data
  userMovies, userActors, userDirectors = user

  cos_sim = [ cosine(m, userMovies) for m in movies ]
  cos_sim = np.array(cos_sim)

  actor_sig = 0.10
  act_sim = []
  for a in actors:
    if np.linalg.norm(a) != 0:
      act_sim.append( cosine(a, userActors) )
    else:
      act_sim.append( 0 )
  act_sim = np.array(act_sim)
  
  director_sig = 0.15
  director_sim = []
  for d in directors: 
    if np.linalg.norm(d) != 0:
      director_sim.append( cosine(d, userDirectors) )
    else:
      director_sim.append( 0 )
  director_sim = np.array(director_sim)

  sim = cos_sim * (1-actor_sig-director_sig)
  sim = sim + (actor_sig * act_sim)
  sim = sim + (director_sig * director_sim)

  arg_max = np.flip( np.argsort(sim) )
  ret = [ names[i] for i in arg_max ]
  return ret[:num]


if __name__ == '__main__':
  main()