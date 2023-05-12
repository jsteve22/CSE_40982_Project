#!/usr/bin/env python3

import numpy as np
import pandas as pd
from MicroPlanner import MicroPlanner
from Messages import Question, AskSimilar

YES = ['yes', 'yes i would like that', 'yeah', 'yes sir', 'yes please', 'yea', 'yes i want that']
NO  = ['no', 'not feeling that', 'i do not want that', 'no sir', 'no thank you', 'not for me', 'nah']
FILE = 'questions.csv'

class DocumentPlanner():

  def __init__(self):
    self.messages = []
    self.mp = MicroPlanner()
    popular = pd.read_pickle('./popular_movies.pkl')
    popular = popular.reset_index(drop=True)
    self.movies = popular[popular['numVotes'] > 50000].sort_values(by=['numVotes','averageRating','startYear'], ascending=False).iloc[:250]
  
  def ask_greeting(self):
    q = Question(0, FILE)
    return self.mp.lexicalize(q)

  def ask_popular(self):
    q = Question(3, FILE)
    return self.mp.lexicalize(q)
  
  def ask_year(self):
    q = Question(4, FILE)
    return self.mp.lexicalize(q)
  
  def ask_runtime(self):
    q = Question(5, FILE)
    return self.mp.lexicalize(q)
  
  def ask_similar(self, userSaw):
    randomMovieIndex = np.random.randint(0,250)
    randomMovie = self.movies.iloc[randomMovieIndex]
    randomMovieName = randomMovie['primaryTitle']
    while randomMovieName in userSaw:
      randomMovieIndex = np.random.randint(0,250)
      randomMovie = self.movies.iloc[randomMovieIndex]
      randomMovieName = randomMovie['primaryTitle']
    q = AskSimilar(randomMovieName)
    return self.mp.lexicalize(q), randomMovieName
    
  