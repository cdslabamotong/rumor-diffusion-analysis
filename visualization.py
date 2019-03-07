# -*- coding: utf-8 -*-
# Author: Chen Ling

import data_helper as dh
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


def graph_nodes_edges(graph_list):
    edges = []
    individual_nodes = []
    
    for i in range(len(graph_list)):
        u = graph_list.iloc[i]['user_ID']
        if graph_list.iloc[i]['influencing_user_ID'] != None:
            v = graph_list.iloc[i]['influencing_user_ID']
            topic = {'topic': graph_list.iloc[i]['topics']}
            edges.append((u, v, topic))
        else:
            individual_nodes.append(u)
    return edges, individual_nodes


def generate_graphs(edges, individual_nodes):
    # Define a empty graph
    G = nx.DiGraph()
    
    # Add edges to the graph
    G.add_edges_from(edges)
    G.add_nodes_from(individual_nodes)
    return G

graph_data_may = dh.load_json('graph files/graph_snopes_may.json')
graph_data_june = dh.load_json('graph files/graph_snopes_june.json')
graph_data_july = dh.load_json('graph files/graph_snopes_july.json')

topics_may = dh.get_claim_dict(graph_data_may)
topics_june = dh.get_claim_dict(graph_data_june)
topics_july = dh.get_claim_dict(graph_data_july)

graph_data_may, topics_may = dh.drop_no_claim_rows(graph_data_may, topics_may)
graph_data_june, topics_june = dh.drop_no_claim_rows(graph_data_june, topics_june)
graph_data_july, topics_july = dh.drop_no_claim_rows(graph_data_july, topics_july)

# Getting a list of topics with no duplicates
topics_may_list = list(set(list(graph_data_may.topics)))
topics_june_list = list(set(list(graph_data_june.topics)))
topics_july_list = list(set(list(graph_data_july.topics)))

# Remove duplicate dicts from a list
total_topics = list(graph_data_may.topics) + list(graph_data_june.topics) + list(graph_data_july.topics)
total_topics_claims = dh.list_of_dicts_integration(
        [dict(t) for t in {tuple(d.items()) for d in (topics_may + topics_june + topics_july)}])

# Get sorted claim list from the most popular to unpopular
sorted_claims_list = [i[0] for i in Counter(total_topics).most_common()]

sorted_titles_list = [dh.get_title(index, x) for index, x in enumerate(sorted_claims_list)]
total_topics_titles = dict(zip(sorted_claims_list, sorted_titles_list))


graph_data = graph_data_may.append(
        graph_data_june, ignore_index=True).append(
                graph_data_july, ignore_index=True)


'''
list_A = []
for i in range(len(sorted_claims_list[:1000])):
    temp = []
    for j in range(len(graph_data)):
        if graph_data.iloc[j].topics == sorted_claims_list[i]:
            temp.append(graph_data.iloc[j].tweet_ID)
    list_A.append(temp)
    print('{}, Done'.format(i))
'''


'''
user_id = list(graph_data.user_ID)
influencing_user_ID = list(graph_data.influencing_user_ID)

topics = list(graph_data.topics)
topics_counter = list(Counter(topics).most_common())
available_topics = [i[0] for i in topics_counter]
available_claims = []
    
for i in range(len(available_topics)):
    try:
        available_claims.append(dh.get_claim(available_topics[i]))
        print("{}, Done".format(i))
    except:
        available_claims.append(None)
        print("{}, Error".format(i))

edges, individual_nodes = graph_nodes_edges(graph_data)

G_snopes = generate_graphs(edges, individual_nodes)
'''


''' 
# Load graph information
graphs_07 = dh.load_graph_json('twitter_data/graph-2016-07/')
graphs_08 = dh.load_graph_json('twitter_data/graph-2016-08/')
graphs_09 = dh.load_graph_json('twitter_data/graph-2016-09/')
graphs_10 = dh.load_graph_json('twitter_data/graph-2016-10/')
graphs_11 = dh.load_graph_json('twitter_data/graph-2016-11/')
graphs_12 = dh.load_graph_json('twitter_data/graph-2016-12/')
graphs_01 = dh.load_graph_json('twitter_data/graph-2017-01/')

# load all edges into a list
edges_07 = graph_edges(graphs_07)
edges_08 = graph_edges(graphs_08)
edges_09 = graph_edges(graphs_09)
edges_10 = graph_edges(graphs_10)
edges_11 = graph_edges(graphs_11)
edges_12 = graph_edges(graphs_12)
edges_01 = graph_edges(graphs_01)

edges = edges_07 + edges_08 + edges_09 + edges_10 + edges_11 + edges_12 + edges_01

G_07 = generate_graphs(edges_07)
G_08 = generate_graphs(edges_08)
G_09 = generate_graphs(edges_09)
G_10 = generate_graphs(edges_10)
G_11 = generate_graphs(edges_11)
G_12 = generate_graphs(edges_12)
G_01 = generate_graphs(edges_01)

G_overall = generate_graphs(edges)
'''