# -*- coding: utf-8 -*-

import data_helper as dh
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from collections import Counter

topics_time = pd.read_csv('topics_time.csv')
snopes_topics = list(topics_time.topics)
time = [dh.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in list(topics_time.time)]
snopes_time = [dh.get_publish_time(i, x) for i, x in enumerate(snopes_topics)]

veracity = [dh.get_rating(i) for i in snopes_topics]
category = [dh.get_category(i, x) for i, x in enumerate(snopes_topics)]

for i in range(len(veracity)):
    if veracity[i] == 'Unproven' or veracity[i] == 'Miscaptioned' or veracity[i] == 'Mostly False' or veracity[i] == 'Mostly True':
        veracity[i] = 'Mixture'
veracity[182] = 'True'


for i in range(len(snopes_time)):
    if snopes_time[i] is None:
        snopes_time[i] = time[i].replace(hour=0, minute=0, second=0, microsecond=0)

created_time = [dh.parse_datetime(i) for i in list(dh.load_json('dataset/57.json').created_at)]

seq, tweets_per_day = dh.number_of_tweets_per_day(created_time)

seq = [dh.datetime.strptime(i, '%m-%d-%Y') for i in seq]

for i in range(len(seq)):
    if seq[i] >= snopes_time[169]:
        temp = i
        break
    
y = np.array(tweets_per_day[temp:])


data_list = []
start = dh.datetime.strptime('Apr-01-2017', '%b-%d-%Y')
end = dh.datetime.strptime('Sep-01-2017', '%b-%d-%Y')
for i in range(216):
    dataset = dh.load_json('titles_tweets/{}.json'.format(i))
    time_list = [dh.parse_datetime(j) for j in list(dataset.created_at)]
    if dh.time_in_range(start, end, snopes_time[i]):
        before, after = dh.compare_datetime('titles_tweets/{}.json'.format(i), snopes_time[i])
        if after > 0 and before > 0:
            data_list.append(i)
            print("{}, Done".format(i))
        else:
            print("{}, Fail".format(i))


#for i, x in enumerate(data_list):
before_counter, after_counter = [], []
for i, x in enumerate(data_list):
    dataset = dh.load_json('titles_tweets/{}.json'.format(x))
    intervention_time = snopes_time[x] + dh.timedelta(days=1)
    time_list = [dh.parse_datetime(i) for i in list(dataset.created_at)]
    
    ind = 0
    while time_list[ind] >= intervention_time:
        ind += 1
        
    before_dataset = dataset.iloc[ind:, :]
    after_dataset = dataset.iloc[:ind, :]
    
    before_counter.append(dh.number_of_retweets(before_dataset))
    after_counter.append(dh.number_of_retweets(after_dataset))
    
   
    
before_counter, after_counter, before_average_size, after_average_size = [], [], [], []
for i, x in enumerate(data_list):
    dataset = dh.load_json('titles_tweets/{}.json'.format(x))
    dataset = dh.resort_dataframe(dataset)
    intervention_time = snopes_time[x] + dh.timedelta(days=1)
    time_list = [dh.parse_datetime(i) for i in list(dataset.created_at)]
    
    ind = 0
    while time_list[ind] >= intervention_time:
        ind += 1
    
    before_dataset = dataset.iloc[ind:, :]
    after_dataset = dataset.iloc[:ind, :]
    
    before_cascades_size = dh.number_of_cascades(before_dataset)
    after_cascades_size = dh.number_of_cascades(after_dataset)

    before_counter.append(len(before_cascades_size))
    before_average_size.append(dh.recursive_len(before_cascades_size)/len(before_cascades_size))
    after_counter.append(len(after_cascades_size))
    after_average_size.append(dh.recursive_len(after_cascades_size)/len(after_cascades_size))
    
    
cascades_evolution_dataset = dh.load_json('dataset/57.json')
cascades_evolution_dataset = dh.resort_dataframe(cascades_evolution_dataset)
evolution_time_list = [dh.parse_datetime(i) for i in list(cascades_evolution_dataset.created_at)]

cascade_number_list = []
cascade_size_list = []

cascade_list = []
for g, df in cascades_evolution_dataset.groupby(np.arange(len(cascades_evolution_dataset)) // 43):
    cascade_list = dh.number_of_cascades(df, cascade_list)
    cascade_number_list.append(len(cascade_list))
    cascade_size_list.append(dh.recursive_len(cascade_list)/len(cascade_list))


''' Cascade Evolution Graph'''
x = list(range(len(cascade_number_list)))

fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(211)
plt.plot(x, cascade_number_list, 'g:', linewidth=4, label='The Change of Cascades\' Number Over Time')
ax1.axvline(x=16, c='r', linewidth=4, label='Snopes involving time') 
ax1.axis([None, None, 0, 800])
ax1.legend(loc=2, fontsize=13)
ax1.set_ylabel('Average Cascades\' Number', fontsize=15)
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(True)

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(x, cascade_size_list, 'b:', linewidth=4, label='The Change of Cascades\' Size Over Time')
ax2.axvline(x=16, c='r', linewidth=4, label='Snopes involving time') 
ax2.axis([None, None, 1, 2.4])
ax2.legend(loc=2, fontsize=13)
ax2.set_ylabel('Average Cascades\' Size', fontsize=15)

plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(True)
plt.xlabel('Evolution Time', fontsize=20)
plt.savefig('paper/figures/figure_6.png', dpi=500)
plt.show()


'''Category Distribution Graph'''








