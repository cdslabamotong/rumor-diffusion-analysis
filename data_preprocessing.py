# -*- coding: utf-8 -*-
# Author: Chen Ling

__author__ = 'Chen'

import pandas as pd
import data_helpers as dh
from tokens import *
from textblob import TextBlob 
import re

file_path = 'twitter_data/status-2016-07/'


def clean_tweet(tweet): 
    ''' 
    Utility function to clean tweet text by removing links, special characters 
    using simple regex statements. 
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 
  
    
def get_tweet_sentiment(tweet): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(clean_tweet(tweet)) 
    # set sentiment 
    if analysis.sentiment.polarity > 0: 
        return 'positive'
    elif analysis.sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'


def data_loadup(names, file_path):
    tweets = []
    for i in names:
        tweets.append(dh.load_json(file_path + i))

    tweets = [item for sublist in tweets for item in sublist]
    return tweets

def load_data():
    tweet_list = []
    file_path_07 = 'twitter_data/status-2016-07/'; file_path_08 = 'twitter_data/status-2016-08/'
    file_path_09 = 'twitter_data/status-2016-09/'; file_path_10 = 'twitter_data/status-2016-10/'
    file_path_11 = 'twitter_data/status-2016-11/'; file_path_12 = 'twitter_data/status-2016-12/'
    file_path_01 = 'twitter_data/status-2017-01/'

    names_2016_07 = dh.filename_in_folder(file_path_07)
    names_2016_08 = dh.filename_in_folder(file_path_08)
    names_2016_09 = dh.filename_in_folder(file_path_09)
    names_2016_10 = dh.filename_in_folder(file_path_10)
    names_2016_11 = dh.filename_in_folder(file_path_11)
    names_2016_12 = dh.filename_in_folder(file_path_12)
    names_2017_01 = dh.filename_in_folder(file_path_01)


    tweets_07 = data_loadup(names_2016_07, file_path_07)
    tweets_08 = data_loadup(names_2016_08, file_path_08)
    tweets_09 = data_loadup(names_2016_09, file_path_09)
    tweets_10 = data_loadup(names_2016_10, file_path_10)
    tweets_11 = data_loadup(names_2016_11, file_path_11)
    tweets_12 = data_loadup(names_2016_12, file_path_12)
    tweets_01 = data_loadup(names_2017_01, file_path_01)

    tweet_list = tweets_07 + tweets_08 + tweets_09 + tweets_10 + tweets_11 + tweets_12 + tweets_01
    
    return tweet_list

def main():
    tweet_list = load_data()
    pos = []; neu = []; neg = []
    
    for i in range(len(tweet_list)):
        if (get_tweet_sentiment(tweet_list[i]['text'])) == 'positive':
            pos.append(tweet_list[i])
        elif (get_tweet_sentiment(tweet_list[i]['text'])) == 'neutral':
            neu.append(tweet_list[i])
        else:
            neg.append(tweet_list[i])
    
    print('The percentage of supportive tweets is {}%'.format(len(pos)/len(tweet_list)*100))
    print('The percentage of neutral tweets is {}%'.format(len(neu)/len(tweet_list)*100))
    print('The percentage of against tweets is {}%'.format(len(neg)/len(tweet_list)*100))

    
if __author__ == 'Chen':
    main()
