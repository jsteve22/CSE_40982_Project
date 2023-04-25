#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle as pkl

from DocumentPlanner import DocumentPlanner

YES = ['yes', 'yes i would like that', 'yeah', 'yes thank you', 'yes please', 'yea', 'yes i want that']
NO  = ['no', 'not feeling that', 'i do not want that', 'no thanks', 'no thank you', 'not for me', 'nah']

LIKE = ['i like this movie', 'yeah i like this one', 'i enjoyed watching this', 'this one is my favorite', 'i really liked this one']
DISLIKE = ['i do not enjoy this movie', 'i hated this movie', 'this movie sucked', 'i really hated this one', 'i did not like this one']
AMBIG = ['i do not know this one', 'i am unfamiliar with this one', 'i have never heard of this', 'i have not seen this one']

get_prompt = lambda arr: arr[ np.random.randint(0, len(arr)-1) ]
# get_no  = lambda:  NO[ np.random.randint(0, len(NO)-1) ]

def pkl_load(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)

def makeRecommendation(user, data, userSaw, num=10):
   
    names, movies, actors, directors = data
    userMovies, userActors, userDirectors = user

    cosine = lambda x,y: np.dot(x,y) / ( np.linalg.norm(x)*np.linalg.norm(y)) if (np.linalg.norm(x)*np.linalg.norm(y)) != 0 else 0
    
    cos_sim = [cosine(v, userMovies) for v in movies]
    cos_sim = np.array(cos_sim)

    # breakpoint()
    actor_sig = 0.1
    act_sim = []
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

    dp = DocumentPlanner()

    # Question 1: Introduction
    # print(questions.loc[questions['Category'] == 0]['Question'].sample().iloc[0])
    print( dp.ask_greeting() )
    print(get_prompt(YES))
    print()

    # Question 2: Rating
    # print(questions.loc[questions['Category'] == 2]['Question'].sample().iloc[0])

    # answer = np.random.choice(['yes','no'])

    # if answer == 'yes':
    #     userMovies[0] += 0.5

    # print(answer)


    # Question 3: Popular
    # print(questions.loc[questions['Category'] == 3]['Question'].sample().iloc[0])
    print( dp.ask_popular() )

    answer = np.random.choice(['yes','no'])

    if answer == 'yes':
        print( get_prompt(YES) )
        userMovies[1] += 0.35
    else:
        print( get_prompt(NO) )
        userMovies[1] -= 0.35
    print()


    # Question 4: Year filmed
    # print(questions.loc[questions['Category'] == 4]['Question'].sample().iloc[0])
    print( dp.ask_year() )
    answer = np.random.choice(['yes','no'])
    if answer == 'yes':
        print( get_prompt(YES) )
        userMovies[3] -= 0.35
    else:
        print( get_prompt(NO) )
        userMovies[3] += 0.35
    print()


    # Question 5: Runtime
    # print(questions.loc[questions['Category'] == 5]['Question'].sample().iloc[0])
    print( dp.ask_runtime() )
    answer = np.random.choice(['yes','no'])
    if answer == 'yes':
        print( get_prompt(YES) )
        userMovies[2] += 0.35
    else:
        print( get_prompt(NO) )
        userMovies[2] -= 0.35
    print()

    # Question 6: Genres
    # print(questions.loc[questions['Category'] == 1]['Question'].sample().iloc[0])
     
    # answer = np.random.choice(['horror','action','comedy','drama'])    


    # Question 7: Actors 
    # print(questions.loc[questions['Category'] == 6]['Question'].sample().iloc[0])

    # print(actors)


    # Question 8: Directors
    # print(questions.loc[questions['Category'] == 7]['Question'].sample().iloc[0])


    # Question 9: Similar movies
    # popular = pd.read_pickle('../popular_movies.pkl')
    # popular = popular.reset_index(drop=True)

    # top_250 = popular[popular['numVotes'] > 50000].sort_values(by=['numVotes','averageRating','startYear'], ascending=False).iloc[:250]

    userSaw = []
    diff = lambda x, y: (y-x)/2
    bound = lambda array: np.minimum( np.ones(len(array)), np.maximum( np.zeros(len(array)), array) )

    i = 0
    while (i < 3):
        '''
        randomMovieIndex = np.random.randint(0,250)
        randomMovie = top_250.iloc[randomMovieIndex]
        randomMovieName = randomMovie['primaryTitle']
        if randomMovieName in userSaw:
            continue

        userSaw.append(randomMovieName)
        index = names.index(randomMovieName)
        print(f'What do you think about {randomMovieName}?')
        print(f'\t1 if you like it')
        print(f'\t2 if you do not like it')
        print(f'\t3 if you do not know it')
        '''

        q, randomMovieName = dp.ask_similar(userSaw)
        print(q)
        userSaw.append(randomMovieName)
        index = names.index(randomMovieName)

        userInput = np.random.choice([1,2,3])
        ratio = 0.75
        invRatio = 1 - ratio

        # print(userInput)

        if (userInput == 1):
            print( get_prompt(LIKE) )
            userMovies    = userMovies + (ratio * diff(userMovies, movies[index]))
            userActors    = userActors + (ratio * diff(userActors, actors[index]))
            userDirectors = userDirectors + (ratio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        elif (userInput == 2):
            print( get_prompt(DISLIKE) )
            userMovies    = userMovies - (ratio * diff(userMovies, movies[index]))
            userActors    = userActors - (ratio * diff(userActors, actors[index]))
            userDirectors = userDirectors - (ratio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        else:
            print( get_prompt(AMBIG) )
            userMovies    = userMovies - (invRatio * diff(userMovies, movies[index]))
            userActors    = userActors - (invRatio * diff(userActors, actors[index]))
            userDirectors = userDirectors - (invRatio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        print()
          
    userMovies = bound(userMovies)
    userActors = bound(userActors)
    userDirectors = bound(userDirectors)

    user = (userMovies, userActors, userDirectors)
    data = (names, movies, actors, directors)

    want_to_watch = False
    recommendedMovie = ''

    for recommendedMovie in makeRecommendation(user, data, userSaw):
        print(f'Do you want to watch {recommendedMovie} ?')
        want_to_watch = np.random.choice([True, False], p=[0.8,0.2])
        if (want_to_watch):
            print( get_prompt(YES) )
            print()
            # print('yes')
            break
        else:
            print( get_prompt(NO) )
            print()
            # print('no')
    
    '''
    for rm in makeRecommendation(user, data, userSaw, num=30):
        print(rm)
    '''
