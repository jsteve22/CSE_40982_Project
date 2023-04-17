import numpy as np
import pandas as pd
import pickle as pkl

cosine = lambda x,y: np.dot(x,y) / ( np.linalg.norm(x)*np.linalg.norm(y)) if (np.linalg.norm(x)*np.linalg.norm(y)) != 0 else 0

def main():
  f = open('vectors.pkl','rb')
  namesVectors = pkl.load(f)
  f.close()

  f = open('actorsVectors.pkl', 'rb')
  actors = pkl.load(f)
  f.close()

  f = open('directorsVectors.pkl', 'rb')
  directors = pkl.load(f)
  f.close()

  names = [n for n, _ in namesVectors]
  vecs = [v for _, v in namesVectors]

  for movie in topSimilar('Pulp Fiction', names, vecs, actors=actors, directors=directors, num=20):
    print(movie)

  pass

def topSimilar(name, names, vecs, actors=None, directors=None, num=10):
  movieIndex = names.index(name)
  cos_sim = [ cosine(v, vecs[movieIndex]) for v in vecs ]
  cos_sim = np.array(cos_sim)

  actor_sig = 0
  director_sig = 0
  act_sim = np.zeros( len(names) )
  director_sim = np.zeros( len(names) )

  if actors:
    actor_sig = 0.10
    act_sim = []
    for a in actors:
      if np.linalg.norm(a) != 0:
        act_sim.append( cosine(a, actors[movieIndex]) )
      else:
        act_sim.append( 0 )
    # act_sim = [ cosine(a, actors[movieIndex]) for a in actors ]
    act_sim = np.array(act_sim)
  
  if directors:
    director_sig = 0.15
    director_sim = []
    for d in directors: 
      if np.linalg.norm(d) != 0:
        director_sim.append( cosine(a, actors[movieIndex]) )
      else:
        director_sim.append( 0 )
    director_sim = np.array(director_sim)

  cos_sim = cos_sim * (1-actor_sig-director_sig)
  cos_sim = cos_sim + (actor_sig * act_sim)
  cos_sim = cos_sim + (director_sig * director_sim)

  arg_max = np.flip( np.argsort(cos_sim) )
  ret = [ names[i] for i in arg_max ]
  return ret[:num]

if __name__ == '__main__':
  main()