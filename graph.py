# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import data_helper as dh
import pandas as pd
import numpy as np
import os
import networkx as nx



topics_time = pd.read_csv('topics_time.csv')
snopes_topics = list(topics_time.topics)
time = [dh.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in list(topics_time.time)]
snopes_time = [dh.get_publish_time(i, x) for i, x in enumerate(snopes_topics)]

veracity = [dh.get_rating(i) for i in snopes_topics]
category = [dh.get_category(i, x) for i, x in enumerate(snopes_topics)]


def generate_graph_file(dataset, filename):
    user_ID = list(dataset.id_str)
    tweet_ID = [i['id_str'] for i in list(dataset.user)]
    created_time = [dh.parse_datetime(i).strftime("%Y-%m-%d %H:%M:%S") for i in list(dataset.created_at)]
    influencing_user_id = [dh.get_influencing_user_id(i) for index, i in dataset.iterrows()]
    influencing_tweet_id = [dh.get_influencing_tweets_id(i) for index, i in dataset.iterrows()]
    actions = [dh.get_user_actions(i) for index, i in dataset.iterrows()]
    
    dh.convert_json(filename, user_ID, tweet_ID, created_time, influencing_user_id, influencing_tweet_id, actions)


def graph_nodes_edges(graph_list):
    edges = []
    individual_nodes = []
    
    for i in range(len(graph_list)):
        u = graph_list.iloc[i]['user_ID']
        if graph_list.iloc[i]['influencing_user_ID'] != None:
            v = graph_list.iloc[i]['influencing_user_ID']
            edges.append((u, v))
        else:
            individual_nodes.append(u)
    return edges, individual_nodes


def generate_graphs(edges, individual_nodes):
    # Define a empty graph
    G = nx.Graph()
    
    # Add edges to the graph
    G.add_edges_from(edges)
    G.add_nodes_from(individual_nodes)
    return G


for i in range(len(veracity)):
    if veracity[i] == 'Unproven' or veracity[i] == 'Miscaptioned' or veracity[i] == 'Mostly False' or veracity[i] == 'Mostly True':
        veracity[i] = 'Mixture'
veracity[182] = 'True'


for i in range(len(snopes_time)):
    if snopes_time[i] is None:
        snopes_time[i] = time[i].replace(hour=0, minute=0, second=0, microsecond=0)



dataset_graph = dh.load_json('titles_tweets/57.json')
dataset_graph = dh.resort_dataframe(dataset_graph)
intervention_time = snopes_time[169]
time_list = [dh.parse_datetime(i) for i in list(dataset_graph.created_at)]
    
ind = 0
while time_list[ind] < intervention_time:
    ind += 1
    
after_dataset = dataset_graph.iloc[ind:, :]
before_dataset = dataset_graph.iloc[:ind, :]


if not os.path.isfile('graph files/169_before_graph.json'):
    generate_graph_file(before_dataset, 'graph files/169_before_graph')
    
if not os.path.isfile('graph files/169_all_graph.json'):
    generate_graph_file(dataset_graph, 'graph files/169_all_graph')

before_graph_dataset = dh.load_json('graph files/169_before_graph.json')
all_graph_dataset = dh.load_json('graph files/169_all_graph.json')


edges_before, individual_nodes_before = graph_nodes_edges(before_graph_dataset)
G_before = generate_graphs(edges_before, individual_nodes_before)
nx.write_graphml(G_before, "test_before.graphml")


edges_all, individual_nodes_all = graph_nodes_edges(all_graph_dataset)
G_all = generate_graphs(edges_all, individual_nodes_all)
nx.write_graphml(G_all, "test_all.graphml")


