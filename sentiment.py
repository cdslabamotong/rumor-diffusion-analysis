# -*- coding: utf-8 -*-

from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import data_helper as dh


hotel_rev = ['Great place to be when you are in Bangalore.',
             'The place was being renovated when I visited so the seating was limited.',
             'Loved the ambience, loved the food',
             'The food is delicious but not over the top.',
             'Service - Little slow, probably because too many people.',
             'The place is not easy to locate',
             'Mushroom fried rice was tasty']
  
sid = SentimentIntensityAnalyzer()

neg, neu, pos, compound = [], [], [], []

for sentence in hotel_rev:
     ss = sid.polarity_scores(sentence)
     '''
     for k in ss:
         print('{0}: {1}, '.format(k, ss[k]), end='')
     '''
     s = list(ss.values())
     neg.append(s[0])
     neu.append(s[1])
     pos.append(s[2])
     compound.append(s[3])


def date_range(date_list, N):
    date_list = []
    if type(date_list[0]) != dh.datetime:
        start = dh.datetime.strptime(date_list[0],"%Y-%m-%d %H:%M:%S")
        end = dh.datetime.strptime(date_list[-1],"%Y-%m-%d %H:%M:%S")
    else:
        start = min(date_list)
        end = max(date_list)
        
    diff = (end - start)/N
    
    date_list.append(start)
    
    for i in range(N):
        date_list.append(start + diff * i)
    
    date_list.append(end)
    return list(date_list)


def tweet_text_analysis(filename):
    dataset = dh.load_json(filename)
    time = [dh.parse_datetime(i) for i in list(dataset.created_at)]
    time_interval = dh.date_range(time, 15)
    tweets = list(dataset.text)
    
    vector = [[] for _ in range(15)]
    
    for i in range(len(vector)):
        for j, x in enumerate(time):
            if dh.time_in_range(time_interval[i], time_interval[i+1], x):
                vector[i].append(dh.remove_mentions_urls(tweets[i]))
        break
    return vector


def 


















