#!/usr/bin/env python3

import pandas as pd
from gensim.models import Word2Vec
from gensim.models import FastText
import textacy
import re
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 32)
    pool = Pool(cpu_count() -1)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def main():
    df = pd.read_csv('tweets.csv',dtype={'tweet':str})
    df = df.dropna()
    # df = dd.from_pandas(df, npartitions=multiprocessing.cpu_count() -1)
    # df = df.map_partitions((lambda x : clean_text(str(x))), columns=['tweet']).compute(get=get)
    df = parallelize_dataframe(df,clean_text_tweet)
#    df['tweet'] = df['tweet'].swifter.apply(lambda x : clean_text(str(x)))
    df = df.head()
    l_tweets = df['tweet'].tolist()
    l_tweets = [twt.split(' ') for twt in l_tweets]

    # model = Word2Vec(l_tweets, 
    #              min_count=20,   # Ignore words that appear less than this
    #              size=100,      # Dimensionality of word embeddings
    #              workers=2,     # Number of processors (parallelisation)
    #              window=5,      # Context window for words during training
    #              iter=30)   
    # model.save("vectors/word2vec.model")
    # model = Word2Vec.load("vectors/word2vec.model")

    model_ft = FastText(l_tweets, 
                 min_count=200,   # Ignore words that appear less than this
                 size=250,      # Dimensionality of word embeddings
                 workers=cpu_count() -1,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30,
                 sg=1)
    
    model_ft.save("data/fasttext.model")
    model_ft = FastText.load("data/fasttext.model")
    
    import pdb; pdb.set_trace()  # breakpoint 5dbfb660 //

    print('test')

def remove_trailing_hashtags(s) :
    words = s.split(' ')
    if((words[-1]).startswith('#')) :
        words = words[:-1]
        return remove_trailing_hashtags(' '.join(words))
    else :
        return s

def remove_starting_ats(s,ats=[]) :
    words = s.split(' ')
    if((words[0]).startswith('@')) :
        words = words[1:]
        return remove_starting_ats(' '.join(words)) 
    else :
       return s

def remove_trailing_ats(s,ats=[]) :
    words = s.split(' ')
    if((words[-1]).startswith('@')) :
        ats.append(words[-1][1:])
        words = words[:-1]
        return remove_trailing_ats(' '.join(words),ats)
    else :
        return s 

def clean_text_tweet(df):
    df['tweet'] = df['tweet'].apply(clean_text)
    return df

def clean_text(this_row):
    this_row = str(this_row)
    this_row = this_row.replace(r'http\S+', '')
    this_row = remove_starting_ats(this_row)
    this_row = remove_trailing_hashtags(this_row)
    this_row = remove_trailing_ats(this_row)
    this_row = this_row.replace("\n",' ')
    this_row = this_row.replace(u"!",'')
    this_row = this_row.replace('-', ' ')
    this_row = this_row.replace('"','')
    this_row = this_row.replace("'",'')
    this_row = this_row.replace("`",'')
    this_row = this_row.replace("#",'')
    this_row = this_row.replace("@",'')
    this_row = this_row.replace("&",'')
    
    this_row = textacy.preprocess.preprocess_text(this_row,
                                                  fix_unicode=True, no_urls=True, no_emails=True,
                                                  lowercase=True, no_contractions=True,
                                                  no_numbers=False, no_currency_symbols=True, no_punct=True)


    return this_row

if __name__ == '__main__':
    main()