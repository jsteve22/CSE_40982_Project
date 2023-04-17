import pandas as pd
import pickle as pkl
import numpy as np

def main():
  # generate_actorsConvert()
  generate_directorsConvert()
  pass

def read_basics():
  basics_df = pd.read_csv('title.basics.tsv/data.tsv', sep='\t', na_values=[r'\N'], dtype={'tconst': str, 
      'titleType': str, 'primaryTitle': str, 'originalTitle': str, 'genres': str}, 
      converters={'isAdult': lambda x: True if x == 1 else False, 'startYear': lambda x: x if x != None else -1, 
      'endYear': lambda x: x if x != None else -1, 'runtimeMinutes': lambda x: x if x != None else -1})
  print(basics_df)
  basics_df.to_pickle('./basics.pkl')
  pass

def clean_basics():
  basics_df = pd.read_pickle('./basics.pkl')
  movies_basics_df = basics_df[ basics_df['titleType'] == 'movie' ]
  movies_basics_df.to_pickle('./movies_basics.pkl')
  pass

def read_ratings():
  ratings_df = pd.read_csv('title.ratings.tsv/data.tsv', sep='\t', na_values=[r'\N'], dtype={'tconst': str, 
      'averageRating': float, 'numVotes': int}) 
  print(ratings_df)
  ratings_df.to_pickle('./ratings.pkl')
  pass

def read_crew():
  crew_df = pd.read_csv('title.crew.tsv/data.tsv', sep='\t', na_values=[r'\N'], dtype={'tconst': str, 
      'directors': str, 'writers': str}) 
  print(crew_df)
  crew_df.to_pickle('./crew.pkl')
  pass

def read_name():
  name_df = pd.read_csv('name.basics.tsv/data.tsv', sep='\t', na_values=[r'\N'], dtype={'nconst': str, 
      'primaryName': str, 'birthYear': str, 'deathYear': str, 'primaryProfession': str, 'knownForTitles': str}) 
  print(name_df)
  name_df.to_pickle('./name.pkl')
  pass

def read_principals():
  principals_df = pd.read_csv('title.principals.tsv/data.tsv', sep='\t', na_values=[r'\N'], dtype={'tconst': str, 
      'ordering': int, 'nconst': str, 'categor': str, 'job': str, 'characters': str}) 
  print(principals_df)
  principals_df.to_pickle('./principals.pkl')
  pass

def combine_movies_principals():
  princ_df = pd.read_pickle('./principals.pkl')
  movies_df = pd.read_pickle('./movies_basics.pkl')
  print(princ_df)
  print(movies_df)

  left = movies_df.merge(princ_df, how='left', on='tconst')
  left = left.groupby(['tconst','titleType','primaryTitle','originalTitle','isAdult','startYear','endYear','runtimeMinutes','genres']).agg(tuple).applymap(list).reset_index()

  left.to_pickle('./movies_principals.pkl')
  pass

def combine_movies_ratings():
  movies_df = pd.read_pickle('./movies_principals.pkl')
  ratings_df = pd.read_pickle('./ratings.pkl')

  total_df = movies_df.merge(ratings_df, how='left', on='tconst')
  total_df.to_pickle('./movies_ratings.pkl')
  pass

def generate_actorsConvert():
  names = pd.read_pickle('./name.pkl')
  actorsConvert = {}
  for index, row in names.iterrows():
    try:
      name = row['primaryName'].lower()
      nconst = row['nconst']
      professions = row['primaryProfession'].lower().split(',')
      if 'actor' in professions or 'actress' in professions:
        actorsConvert[nconst] = name
    except:
      pass

  with open('actorsConvert.pkl', 'wb') as f:
    pkl.dump(actorsConvert, f)
  return

def generate_directorsConvert():
  names = pd.read_pickle('./name.pkl')
  directorsConvert = {}
  for _, row in names.iterrows():
    try:
      name = row['primaryName'].lower()
      nconst = row['nconst']
      professions = row['primaryProfession'].lower().split(',')
      if 'director' in professions: 
        directorsConvert[nconst] = name
    except:
      pass

  with open('directorsConvert.pkl', 'wb') as f:
    pkl.dump(directorsConvert, f)
  return

if __name__ == '__main__':
  main()