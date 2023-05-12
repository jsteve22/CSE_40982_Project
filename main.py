#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import time

from DocumentPlanner import DocumentPlanner


dp = DocumentPlanner()

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

def categorize_sentence(sentence, model, tokenizer):
    seq = tokenizer.texts_to_sequences( [sentence] )
    newseq = pad_sequences(seq, padding='post', truncating='post', maxlen=200)
    pred = model.predict( newseq )
    '''
    ret = np.argmax(pred)
    if ret == 0:
        print('RESPONSE == YES')
    elif ret == 1:
        print('RESPONSE == NO')
    elif ret == 2:
        print('RESPONSE == AMBIG')
    '''

    return np.argmax(pred)

cosine = lambda x,y: np.dot(x,y) / ( np.linalg.norm(x)*np.linalg.norm(y)) if (np.linalg.norm(x)*np.linalg.norm(y)) != 0 else 0

def makeRecommendation(user, data, userSaw, num=10):
   
    names, movies, actors, directors = data
    userMovies, userActors, userDirectors = user
    
    # cos_sim = [cosine(v, userMovies) for v in movies]
    m_start = time.perf_counter()
    cos_sim = [cosine(v, userMovies) for v in movies]
    m_end = time.perf_counter() - m_start
    cos_sim = np.array(cos_sim)

    # breakpoint()
    actor_sig = 0.1
    a_start = time.perf_counter()
    act_sim = []
    for a in actors:
        if np.linalg.norm(a) != 0:
            act_sim.append(cosine(a, userActors))
        else:
            act_sim.append(0)
    a_end = time.perf_counter() - a_start
    act_sim = np.array(act_sim)

    director_sig = 0.15
    d_start = time.perf_counter()
    director_sim = []
    for d in directors:
        if np.linalg.norm(d) != 0:
            director_sim.append(cosine(d, userDirectors))
        else:
            director_sim.append(0)
    d_end = time.perf_counter() - d_start
    director_sim = np.array(director_sim)

    sim = cos_sim * (1 - actor_sig - director_sig)
    sim += (actor_sig * act_sim)
    sim += (director_sig * director_sim)

    arg_max = np.flip( np.argsort(sim))
    ret = [ names[i] for i in arg_max ] 
    arr = [ movies[i] for i in arg_max ]

    # print(f'times:{m_end:.5f}\t{a_end:0.5f}\t{d_end:0.5f}')
    if userSaw:
        r = []
        a = []
        for ri, ai in zip(ret, arr):
            if ri not in userSaw:
                r.append((ri,ai))
                if len(r) == num:
                    return r

    return ret[0], arr[0]

def main(movieVecs=None, actors=None, directors=None, questions=None):

    # load tokenizer
    token_json = ''
    with open('./movie_tok.json', 'r') as f:
        for line in f:
            token_json += line.rstrip()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json( token_json )
    model = tf.keras.models.load_model('./movie_lstm.h5')

    output = sys.stdout 
    # Load in vectors
    movieVecs = pkl_load('./vectors.pkl') if movieVecs == None else movieVecs 
    actors    = pkl_load('./aVectors.pkl') if actors == None else actors    
    directors = pkl_load('./dVectors.pkl') if directors == None else directors 
    questions = questions
    try:
        if questions == None:
            questions = pd.read_csv('questions.csv',names=['Category','Question'],header=None) 
    except:
        pass

    names  = [n for n, _ in movieVecs]
    movies = [v for _, v in movieVecs]

    userMovies    = np.ones(len(movies[0]), dtype=float) / 2
    userActors    = np.ones(len(actors[0]), dtype=float) / 2
    userDirectors = np.ones(len(directors[0]), dtype=float) / 2

    randomNum = np.random.randint(0,len(movies))
    target = np.array(movies[randomNum])
    # print(f'movie: {names[randomNum]}')
    # print(f'array: {target}')


    # Question 1: Introduction
    # output.write(questions.loc[questions['Category'] == 0]['Question'].sample().iloc[0])
    output.write(f'{dp.ask_greeting()}\n')
    resp = categorize_sentence( input(), model, tokenizer )
    if resp == 1:
        output.write(f'Have a nice day!\n')
        return

    # Question 2: Popular
    # output.write(questions.loc[questions['Category'] == 3]['Question'].sample().iloc[0])
    output.write(f'{dp.ask_popular()}\n')
    resp = categorize_sentence( input(), model, tokenizer )

    if resp == 0:
        userMovies[1] += 0.35
    else:
        userMovies[1] -= 0.35
    output.write('\n')


    # Question 4: Year filmed
    # output.write(questions.loc[questions['Category'] == 4]['Question'].sample().iloc[0])
    output.write(f'{dp.ask_year()}\n')
    resp = categorize_sentence( input(), model, tokenizer )
    if resp == 0:
        userMovies[3] -= 0.35
    else:
        userMovies[3] += 0.35
    output.write('\n')


    # Question 5: Runtime
    # output.write(questions.loc[questions['Category'] == 5]['Question'].sample().iloc[0])
    output.write(f'{dp.ask_runtime()}\n')
    resp = categorize_sentence( input(), model, tokenizer )
    if resp == 0:
        userMovies[2] += 0.35
    else:
        userMovies[2] -= 0.35
    output.write('\n')

    userSaw = []
    diff = lambda x, y: (y-x)/2
    bound = lambda array: np.minimum( np.ones(len(array)), np.maximum( np.zeros(len(array)), array) )

    i = 0
    output.write('Still collecting data...\n')
    while (i < 3):

        q, randomMovieName = dp.ask_similar(userSaw)
        output.write(f'{q}\n')
        userSaw.append(randomMovieName)
        index = names.index(randomMovieName)

        resp = categorize_sentence( input(), model, tokenizer )

        ratio = 0.75
        invRatio = 1 - ratio

        # output.write(userInput)

        if (resp == 0):
            userMovies    = userMovies + (ratio * diff(userMovies, movies[index]))
            userActors    = userActors + (ratio * diff(userActors, actors[index]))
            userDirectors = userDirectors + (ratio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        elif (resp == 1):
            userMovies    = userMovies - (ratio * diff(userMovies, movies[index]))
            userActors    = userActors - (ratio * diff(userActors, actors[index]))
            userDirectors = userDirectors - (ratio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        else:
            userMovies    = userMovies - (invRatio * diff(userMovies, movies[index]))
            userActors    = userActors - (invRatio * diff(userActors, actors[index]))
            userDirectors = userDirectors - (invRatio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        output.write('\n')
          
    userMovies = bound(userMovies)
    userActors = bound(userActors)
    userDirectors = bound(userDirectors)

    user = (userMovies, userActors, userDirectors)
    data = (names, movies, actors, directors)

    want_to_watch = False
    recommendedMovie = ''

    output.write('Making recommendations...\n')
    for recommendedMovie, movieArray in makeRecommendation(user, data, userSaw, num=100):
        output.write(f'Do you want to watch {recommendedMovie} ?\n')
        resp = categorize_sentence( input(), model, tokenizer )

        if (resp == 0):
            output.write('Enjoy your movie!\n')
            break
        else:
            output.write('\n')
    # else:
        # output.write('You did not like any of the 100 movies? Really?\n')
    

def generate_sample(movieVecs=None, actors=None, directors=None, questions=None, output=None):
    if output == None:
        output = sys.stdout 
    # Load in vectors
    movieVecs = pkl_load('./vectors.pkl') if movieVecs == None else movieVecs 
    actors    = pkl_load('./aVectors.pkl') if actors == None else actors    
    directors = pkl_load('./dVectors.pkl') if directors == None else directors 
    questions = questions
    try:
        if questions == None:
            questions = pd.read_csv('questions.csv',names=['Category','Question'],header=None) 
    except:
        pass

    names  = [n for n, _ in movieVecs]
    movies = [v for _, v in movieVecs]

    userMovies    = np.ones(len(movies[0]), dtype=float) / 2
    userActors    = np.ones(len(actors[0]), dtype=float) / 2
    userDirectors = np.ones(len(directors[0]), dtype=float) / 2

    randomNum = np.random.randint(0,len(movies))
    target = np.array(movies[randomNum])
    # print(f'movie: {names[randomNum]}')
    # print(f'array: {target}')


    # Question 1: Introduction
    # output.write(questions.loc[questions['Category'] == 0]['Question'].sample().iloc[0])
    output.write(f'QUESTION:GREETING:: {dp.ask_greeting()}\n')
    output.write(f'ANSWER:YES:: {get_prompt(YES)}\n')
    output.write('\n')

    # Question 2: Popular
    # output.write(questions.loc[questions['Category'] == 3]['Question'].sample().iloc[0])
    output.write(f'QUESTION:POPULAR:: {dp.ask_popular()}\n')

    if target[1] > 0.5:
        output.write(f'ANSWER:YES:: {get_prompt(YES)}\n')
        userMovies[1] += 0.35
    else:
        output.write(f'ANSWER:NO:: {get_prompt(NO)}\n')
        userMovies[1] -= 0.35
    output.write('\n')


    # Question 4: Year filmed
    # output.write(questions.loc[questions['Category'] == 4]['Question'].sample().iloc[0])
    output.write(f'QUESTION:YEAR:: {dp.ask_year()}\n')
    answer = np.random.choice(['yes','no'])
    if target[3] < 0.5:
        output.write(f'ANSWER:YES:: {get_prompt(YES)}\n')
        userMovies[3] -= 0.35
    else:
        output.write(f'ANSWER:NO:: {get_prompt(NO)}\n')
        userMovies[3] += 0.35
    output.write('\n')


    # Question 5: Runtime
    # output.write(questions.loc[questions['Category'] == 5]['Question'].sample().iloc[0])
    output.write(f'QUESTION:RUNTIME:: {dp.ask_runtime()}\n')
    answer = np.random.choice(['yes','no'])
    if target[2] > 0.5:
        output.write(f'ANSWER:YES:: {get_prompt(YES)}\n')
        userMovies[2] += 0.35
    else:
        output.write(f'ANSWER:NO:: {get_prompt(NO)}\n')
        userMovies[2] -= 0.35
    output.write('\n')

    # Question 6: Genres
    # output.write(questions.loc[questions['Category'] == 1]['Question'].sample().iloc[0])
     
    # answer = np.random.choice(['horror','action','comedy','drama'])    

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
        output.write(f'What do you think about {randomMovieName}?')
        output.write(f'\t1 if you like it')
        output.write(f'\t2 if you do not like it')
        output.write(f'\t3 if you do not know it')
        '''

        q, randomMovieName = dp.ask_similar(userSaw)
        output.write(f'QUESTION:SIMILAR:: {q}\n')
        userSaw.append(randomMovieName)
        index = names.index(randomMovieName)

        # userInput = np.random.choice([1,2,3])
        userInput = 3
        cosineSim = cosine(target, np.array(movies[index]))
        # print(f'cosineSim in while: {cosineSim}')
        if (cosineSim > 0.60):
            userInput = 1
        if (cosineSim < 0.40):
            userInput = 2

        ratio = 0.75
        invRatio = 1 - ratio

        # output.write(userInput)

        if (userInput == 1):
            output.write(f'ANSWER:YES:: {get_prompt(LIKE)}\n')
            userMovies    = userMovies + (ratio * diff(userMovies, movies[index]))
            userActors    = userActors + (ratio * diff(userActors, actors[index]))
            userDirectors = userDirectors + (ratio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        elif (userInput == 2):
            output.write(f'ANSWER:NO:: {get_prompt(DISLIKE)}\n')
            userMovies    = userMovies - (ratio * diff(userMovies, movies[index]))
            userActors    = userActors - (ratio * diff(userActors, actors[index]))
            userDirectors = userDirectors - (ratio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        else:
            output.write(f'ANSWER:AMBIG:: {get_prompt(AMBIG)}\n')
            userMovies    = userMovies - (invRatio * diff(userMovies, movies[index]))
            userActors    = userActors - (invRatio * diff(userActors, actors[index]))
            userDirectors = userDirectors - (invRatio * diff(userDirectors, directors[index]))
            i += 1
            ratio = ratio ** 2
        output.write('\n')
          
    userMovies = bound(userMovies)
    userActors = bound(userActors)
    userDirectors = bound(userDirectors)

    user = (userMovies, userActors, userDirectors)
    data = (names, movies, actors, directors)

    want_to_watch = False
    recommendedMovie = ''

    for recommendedMovie, movieArray in makeRecommendation(user, data, userSaw):
        output.write(f'QUESTION:RECOMMENDATION:: Do you want to watch {recommendedMovie} ?\n')
        # want_to_watch = np.random.choice([True, False], p=[0.8,0.2])
        cosineSim = cosine(target, np.array(movieArray))
        # print(f'consineSim in recommend: {cosineSim}')
        want_to_watch = True
        simLimit = 0.50
        if cosineSim < simLimit:
            want_to_watch = False
        simLimit -= 0.10

        if (want_to_watch):
            output.write(f'ANSWER:YES:: {get_prompt(YES)}\n')
            output.write('\n')
            # output.write('yes')
            break
        else:
            output.write(f'ANSWER:NO:: {get_prompt(NO)}\n')
            output.write('\n')
            # output.write('no')
    
    '''
    for rm in makeRecommendation(user, data, userSaw, num=30):
        output.write(rm)
    '''

if __name__ == '__main__':
    movieVecs = pkl_load('./vectors.pkl') 
    actors    = pkl_load('./aVectors.pkl') 
    directors = pkl_load('./dVectors.pkl') 
    questions = pd.read_csv('questions.csv',names=['Category','Question'],header=None) 
    main(movieVecs, actors, directors, questions)
    '''
    nextStart = 7000
    start = 0 + nextStart
    end = 3000 + nextStart
    for i in range(start, end):
        with open(f'samples/{i}.txt', 'w') as f:
            generate_sample(movieVecs, actors, directors, questions, output=f)
    '''