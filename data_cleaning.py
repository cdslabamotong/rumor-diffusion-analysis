# -*- coding: utf-8 -*-

import data_helper as dh
from tokens import *

'''
graph_data_may = dh.load_json('graph files/graph_snopes_may.json')
graph_data_june = dh.load_json('graph files/graph_snopes_june.json')
graph_data_july = dh.load_json('graph files/graph_snopes_july.json')

graph_data = graph_data_may.append(
        graph_data_june, ignore_index=True).append(
                graph_data_july, ignore_index=True)

total_topics = list(graph_data_may.topics) + list(graph_data_june.topics) + list(graph_data_july.topics)
sorted_topics_list = [i[0] for i in dh.Counter(total_topics).most_common()]
'''



data_may = dh.load_json('data/snopes_may.json')
data_june = dh.load_json('data/snopes_june.json')
data_july = dh.load_json('data/snopes_july.json')

graph_data = data_may.append(data_june, ignore_index=True).append(data_july, ignore_index=True)
 

for i, x in enumerate(data_list):
    testing = dh.load_json("titles_tweets/{}.json".format(x))
    my_list = dh.read_into_list(x)
    for j in range(len(graph_data)):
        if graph_data.iloc[j]['id_str'] in my_list:
            testing = testing.append(graph_data.iloc[j], ignore_index=True)
             
    testing = testing.drop_duplicates(subset='id_str', keep="first")
    
    testing.to_json('dataset/{}.json'.format(i), orient='records', lines=True)
    
    print("Done, {}".format(i))
         
    
for i, x in enumerate(data_list):
    testing = dh.load_json("dataset/{}.json".format(x))
    my_list = dh.read_into_list(x)
    drop_list = []
    for j in range(len(testing)):
        if testing.iloc[j]['id_str'] in my_list:
            drop_list.append(j)

    testing = testing.drop(testing.index[drop_list])
    
    testing.to_json('dataset_2/{}.json'.format(i), orient='records', lines=True)
    
    print("Done, {}".format(i))
    
    
    