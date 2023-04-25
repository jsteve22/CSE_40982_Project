#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle as pkl

def pkl_load(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)

def makeRecommendation(user, data, userSaw):
   
    names, movies, actors, directors = data
    userMovies, userActors, userDirectors = user

    cosine = lambda x,y: np.dot(x,y) / ( np.linalg.norm(x)*np.linalg.norm(y)) if (np.linalg.norm(x)*np.linalg.norm(y)) != 0 else 0
    
    cos_sim = [cosine(v, userVec) for v in movies]
    cos_sim = np.array(cos_sim)

    actor_sig = 0.1
    actor_sim = []
    for a in actors:
        if np.linalg.norm(a) != 0:
            act_sim.append(cosine(a, userActors))
        else:
            act_sim.append(0)
    act_sim = np.array(act_sim)

    director_sig = 0.15
    director_sim = []
    for d in directors:
        if np.linalg.norm(d) != 0:
            director_sim.append(cosine(d, userDirectors))
        else:
            director_sim.append(0)
    director_sim = np.array(director_sim)

    sim = cos_sim * (1 - actor_sig - director_sig)
    sim += (actor_sig * act_sim)
    sim += (director_sig * director_sim)

    arg_max = np.flip( np.argsort(sim))
    ret = [ names[i] for i in arg_max ] 
    if userSaw:
        r = []
        for ri in ret:
            if ri not in userSaw:
                r.append(ri)
                if len(r) == num:
                    return r

    return ret[0]

if __name__ == '__main__':

    # Load in vectors
    movieVecs = pkl_load('../vectors.pkl')
    actors    = pkl_load('../actorsVectors.pkl')
    directors = pkl_load('../directorsVectors.pkl')

    names  = [n for n, _ in movieVecs]
    movies = [v for _, v in movieVecs]

    userMovies    = np.ones(len(movies[0]), dtype=float) / 2
    userActors    = np.ones(len(actors[0]), dtype=float) / 2
    userDirectors = np.ones(len(directors[0]), dtype=float) / 2

    questions = pd.read_csv('questions.csv',names=['Category','Question'],header=None)

    # Question 1: Introduction
    print(questions.loc[questions['Category'] == 0]['Question'].sample().iloc[0])
   
    print('yes')

    # Question 2: Rating
   # print(questions.loc[questions['Category'] == 2]['Question'].sample().iloc[0])
   
   # answer = np.random.choice(['yes','no'])

   # if answer == 'yes':
   #     userMovies[0] += 0.5

   # print(answer)


    # Question 3: Popular
    print(questions.loc[questions['Category'] == 3]['Question'].sample().iloc[0])

    answer = np.random.choice(['yes','no'])

    if answer == 'yes':
        userMovies[1] += 0.25
    else:
        userMovies[1] -= 0.25

    print(answer)


    # Question 4: Year filmed
    print(questions.loc[questions['Category'] == 4]['Question'].sample().iloc[0])

    answer = np.random.choice(['yes','no'])

    if answer == 'yes':
        userMovies[3] -= 0.25
    else:
        userMovies[3] += 0.25

    print(answer)


    # Question 5: Runtime
    print(questions.loc[questions['Category'] == 5]['Question'].sample().iloc[0])

    answer = np.random.choice(['yes','no'])

    if answer == 'yes':
        userMovies += 0.25
    else:
        userMovies -= 0.25

    print(answer)


    # Question 6: Genres
    # print(questions.loc[questions['Category'] == 1]['Question'].sample().iloc[0])
     
    # answer = np.random.choice(['horror','action','comedy','drama'])    


    # Question 7: Actors 
    # print(questions.loc[questions['Category'] == 6]['Question'].sample().iloc[0])

    # print(actors)



    # Question 8: Directors
    # print(questions.loc[questions['Category'] == 7]['Question'].sample().iloc[0])


    # Question 9: Similar movies
    popular = pd.read_pickle('../popular_movies.pkl')
    popular = popular.reset_index(drop=True)

    top_250 = popular[popular['numVotes'] > 50000].sort_values(by=['numVotes','averageRating','startYear'], ascending=False).iloc[:250]

    userSaw = []

    i = 0
    while (i < 3):
        randomMovieIndex = np.random.randit(0,250)
        randomMovie = top_250.iloc[randomMovieIndex]
        randomMovieName = randomMovie['primaryTitle']
        if randomMovie in userSaw:
            continue

        userSaw.append(randomMovieName)
        index = names.index(randomMovieName)
        print(f'What do you think about {randomMovieName}?')
        print(f'\t1 if you like it')
        print(f'\t2 if you do not like it')
        print(f'\t3 if you do not know it')

        userInput = np.random.choice([1,2,3])
        ratio = 0.75
        invRatio = 1 - ratio

        print(userInput)

        if (userInput == 1):
            userMovies    = min(1, userMovies + (ratio * diff(userMovies, movies[index])))
            userActors    = min(1, userActors + (ratio * diff(userActors, actors[index])))
            userDirectors = min(1, userDirectors + (ratio * diff(userDirectors, directors[index])))
            i += 1
            ratio = ratio ** 2

        elif (userInput == 2):
            userMovies    = max(0, userMovies - (ratio * diff(userMovies, movies[index])))
            userActors    = max(0, userActors - (ratio * diff(userActors, actors[index])))
            userDirectors = max(0, userDirectors - (ratio * diff(userDirectors, directors[index])))
            i += 1
            ratio = ratio ** 2

        else:
            pass
          
    userMovies = bound(userMovies)
    userActors = bound(userActors)
    userDirectors = bound(userDirectors)

    user = (userMovies, userActors, userDirectors)
    data = (names, movies, actors, directors)

    want_to_watch = False
    recommendedMovie = ''

    while not want_to_watch:
        recommendedMovie = makeRecommendation(user, data, userSaw)
    
        print(f'Do you want to watch {recommendedMovie}?')

        watch_to_watch = np.random.choice([True, False], p=[0.8,0.2])

        if watch_to_watch:
            print('yes')
        else:
            print('no')
            userSaw.append(recommendedMovie)
