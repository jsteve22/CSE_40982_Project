#!/usr/bin/env python3

from random import randint

class Message():
  def __init__(self):
    return

class Question(Message):

  def __init__(self, num, csv):
    self.num = num
    self.all_questions = self.load_csv(csv)
    self.questions = self.all_questions[self.num] 
    self.q = self.questions[ randint(0, len(self.questions)-1) ]
  
  def load_csv(self, csv):
    d = {}
    with open(f'{csv}', 'r') as fr:
      for line in fr:
        assert len(line.split(',')) == 2
        ind, q = line.rstrip().split(',')
        ind = int(ind)
        if ind not in d:
          d[ind] = [q]
        else:
          d[ind].append(q)
    return d
  
  def __str__(self):
    return f'{self.q}'
  
  def __repr__(self):
    return f'{self.q}'

class AskSimilar(Message):
  def __init__(self, movie):
    self.s = f'What do you think about {movie} ?'
  
  def __str__(self):
    return f'{self.s}'

  def __repr__(self):
    return f'{self.s}'