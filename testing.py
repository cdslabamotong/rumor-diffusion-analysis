# -*- coding: utf-8 -*-
# Author: chen
# Created at: 12/13/18 10:52 AM

from collections import Counter
import pandas as pd
import data_helper as dh
from searchtweets import gen_rule_payload, load_credentials, collect_results
import requests
import networkx as nx

api = dh.load_api()

def drop_non_en(dataframe):
    non_en_list = []
    for i in range(len(dataframe)):
        if dataframe.iloc[i].lang != 'en':
            non_en_list.append(i)
            
    dataframe = dataframe.drop(non_en_list)
    return dataframe


snopes_dataframe = dh.load_json('data/snopes_july.json')
snopes_dataframe = drop_non_en(snopes_dataframe)
# Extractng urls from the tweets
counter = 0
ids = list(snopes_dataframe.id_str)
urls = list(snopes_dataframe.entities)
extended_urls = list(snopes_dataframe.extended_tweet)


for i in range(len(urls)):
    try:
        urls[i] = urls[i]['urls']
        urls[i] = urls[i][0]
        counter += 1
    except IndexError:
        urls[i] = None


topics = []

for i in range(len(ids)):
    if urls[i] is None:
        if 'twitter.com/' in snopes_dataframe.iloc[i].retweeted_status['entities']['urls'][0]['expanded_url']:
            topics.append(snopes_dataframe.iloc[i].retweeted_status['extended_tweet']['entities']['urls'][0]['expanded_url'])
        else:
            topics.append(snopes_dataframe.iloc[i].retweeted_status['entities']['urls'][0]['expanded_url'])
    else:
        if 'twitter.com/' in urls[i]['expanded_url']:
            if type(snopes_dataframe.iloc[i]['extended_tweet']) is not float:
                topics.append(snopes_dataframe.iloc[i]['extended_tweet']['entities']['urls'][0]['expanded_url'])
            else:
                topics.append(None)
        else:
            topics.append(urls[i]['expanded_url'])
        

errors = []
for i in range(len(topics)):
    if topics[i] is not None:
        if 'archive.is/' not in topics[i]:
            try:
                r = requests.get(topics[i])
                topics[i] = r.url
                print("The {}, done".format(i))
            except requests.exceptions.ConnectionError:
                errors.append(i)
                print("The {}, error".format(i))
                continue


drop_list = []    
for i in range(len(topics)):
    if topics[i] is None:
        drop_list.append(i)
    else:
        if 'fact-check' not in topics[i]:
            drop_list.append(i)
        elif 'fact-check/category/' in topics[i]:
            drop_list.append(i)
        elif 'https://www.snopes.com/fact-check/' == topics[i]:
            drop_list.append(i)
        elif 'https://www.snopes.com/fact-check-ratings/' == topics[i]:
            drop_list.append(i)
            
cleaned_topic = [v for i,v in enumerate(topics) if i not in drop_list]

rating = []
for i in range(len(cleaned_topic)):
    try:
        aa = dh.get_rating(cleaned_topic[i])
        rating.append(aa)
        print("The {}, Done".format(i))
    except :
        rating.append(None)
        print("The {}, Error".format(i))


snopes_dataframe = snopes_dataframe.drop(snopes_dataframe.index[drop_list])


user_ID = list(snopes_dataframe.id_str)
tweet_ID = [i['id_str'] for i in list(snopes_dataframe.user)]
influencing_user_id = [dh.get_influencing_user_id(i) for index, i in snopes_dataframe.iterrows()]
influencing_tweet_id = [dh.get_influencing_tweets_id(i) for index, i in snopes_dataframe.iterrows()]
actions = [dh.get_user_actions(i) for index, i in snopes_dataframe.iterrows()]

dh.convert_json('graph_snopes_july', user_ID, tweet_ID, influencing_user_id, influencing_tweet_id, actions, cleaned_topic, rating)
