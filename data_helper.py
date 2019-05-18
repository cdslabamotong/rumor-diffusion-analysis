# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
import requests
from bs4 import BeautifulSoup
from urllib import request
from rake_nltk import Rake
from urllib.error import HTTPError
import nltk 
import csv
import re
from searchtweets import gen_rule_payload, load_credentials, collect_results
import datetime
from datetime import datetime, timedelta
from email.utils import mktime_tz, parsedate_tz
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import statistics as s
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


file_path = 'twitter_data/ids-2016-07/'
status_path = 'twitter_data/status-2016-07/'
graph_path = 'twitter_data/graph-2016-07/'

myAttrs = {"class":"rating-text"}
stopwords_list = nltk.corpus.stopwords.words('english')

def collect_tweets(QUERY, FROM_DATE, TO_DATE, MAX_RESULTS, RESULTS_PER_CALL = 100):
    """
    Collect tweets by the given API credentials and the list of constraints
    
    Args:
        QUERY: The query build from the keywords extraction function
        FREM_DATE: Collect tweets from this specific time (date)
        TO_DATE: Collect tweets until this specific time (date)
        RESULTS_PER_CALL: Collect tweets (default: 500)
        
    Returns:
        List of tweets 
    """
    premium_search_args = load_credentials("~/.twitter_keys.yaml",
                                       yaml_key='search_tweets_api',
                                       env_overwrite=False)
    
    rule = gen_rule_payload(QUERY,
                            from_date=FROM_DATE,
                            to_date=TO_DATE,
                            results_per_call=RESULTS_PER_CALL)
    
    tweets = collect_results(rule, 
                             max_results=MAX_RESULTS, 
                             result_stream_args=premium_search_args)

    return tweets

# Write into json file
def convert_json(file_name, 
                 user_ID, 
                 tweet_ID,
                 created_time,
                 influencing_user_id, 
                 influencing_tweet_id, 
                 actions):
    dict_list = []
    entities = ['user_ID', 
            'tweet_ID', 
            'create_time',
            'influencing_user_ID', 
            'influencing_tweet_ID',
            'actions']
    
    for i in range(len(user_ID)):
        attributes = []
        attributes.append(user_ID[i])
        attributes.append(tweet_ID[i])
        attributes.append(created_time[i])
        attributes.append(influencing_user_id[i])
        attributes.append(influencing_tweet_id[i])
        attributes.append(actions[i])
        dict_list.append(dict(zip(entities, attributes)))
    
    write_into_json(file_name, dict_list)


def filename_in_folder(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if 'DS_Store' not in filename:
            filenames.append(filename.split('.')[0])
    return filenames


def load_graph_json(graph_path):
    graph = filename_in_folder(graph_path)
    graphs = []
    for i in graph:
        graphs.append(load_json(graph_path + i + '.json'))
    
    flat_graphs = [item for sublist in graphs for item in sublist]
    return flat_graphs


def load_json(filename):
    """
    Load json files in a given directory
    
    Args:
        filename: The given file path with its file name
        
    Returns:
        The pandas dataframe format nd-array
    """
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    data = pd.DataFrame(data)
    return data


def getting_original_url(urls):
    """
    Parsing the original URLs from Tweets status json file
    
    Args:
        urls: The list of all masked urls
        
    Returns:
        All of the shortened urls
    """
    shortened_url = []
    for i in urls:
        if i is not None:
            if i['expanded_url'].endswith('.jpg') is False:
                if 'www.snopes.com/' in i['expanded_url']:
                    shortened_url.append(i['url'])

    for i in range(len(shortened_url)):
        try:
            r = requests.get(shortened_url[i])
            shortened_url[i] = r.url
        except ConnectionError:
            continue
   
    return shortened_url


def get_rating(url):
    """
    Getting the topic rating by a given snopes fact-check url
    
    Args:
        url: The snope fact-check url
        
    Returns:
        The related label
    """
    html = request.urlopen(url)
    bs = BeautifulSoup(html, 'lxml')
    try:
        tag = bs.find_all(name='div', attrs=myAttrs)
        texts = tag[0].text.split('\n')[1]
    except IndexError:
        tag = bs.find_all(name='font', attrs={"class":"status_color"})
        if len(tag) == 0:
            tag = bs.find_all(name='font', attrs={"class":"sect_font_style"})
        texts = tag[0].text
    return texts
    

def get_claim(url):
    """
    Getting the topic's main claim by a given snopes fact-check url
    
    Args:
        url: The snope fact-check url
        
    Returns:
        The related label
    """
    html = request.urlopen(url)
    bs = BeautifulSoup(html, 'lxml')
    try:
        tag = bs.find_all(name='p', attrs={"class":"claim"})
        texts = tag[0].text
    except IndexError:
        tag = bs.find_all(name='h2', attrs={"class":"card-subtitle"})
        texts = tag[0].text
    return texts


def get_title(index, url):
    """
    Getting the topic's title by a given snopes fact-check url
    
    Args:
        url: The snope fact-check url
        
    Returns:
        The related label
    """
    html = request.urlopen(url)
    bs = BeautifulSoup(html, 'lxml')
    try:
        tag = bs.find_all(name='h1', attrs={"class":"card-title"})
        texts = tag[0].text
        print("{}, Done".format(index))
        return texts
    except:
        print("{}, Error".format(index))        
        return None
    

def get_category(index, url):
    """
    Getting the topic's category by a given snopes fact-check url
    
    Args:
        url: The snope fact-check url
        
    Returns:
        The related category
    """
    html = request.urlopen(url)
    bs = BeautifulSoup(html, 'lxml')
    try:
        tag = bs.find_all(name='li', attrs={"class":"breadcrumb-item"})
        texts = tag[1].text
        print("{}, Done".format(index))
        return texts
    except:
        print("{}, Error".format(index))        
        return None
    

def get_publish_time(index, url):
    """
    Getting the topic's title by a given snopes fact-check url
    
    Args:
        url: The snope fact-check url
        
    Returns:
        The related publish time
    """
    html = request.urlopen(url)
    bs = BeautifulSoup(html, 'lxml')
    try:
        tag = bs.find_all(name='span', attrs={"class":"date-published"})
        if tag:
            texts = tag[0].text
            datetime_object = datetime.strptime(texts, '%d %B %Y')
        #else:
        #    tags = bs.find_all(name='span', attrs={"class":"date-updated"})
        #    texts = tags[0].text
        #    datetime_object = datetime.strptime(texts, '%d %B %Y')
            print("{}, Done".format(index))
            return datetime_object
    except IndexError:
        print("{}, Error".format(index))        
        return None


def get_influencing_user_id(dataframe):
    if pd.isnull(dataframe.retweeted_status) is False:
        return dataframe.retweeted_status['user']['id_str']
    elif pd.isnull(dataframe.quoted_status) is False:
        return dataframe.quoted_status['user']['id_str']
    elif pd.isnull(dataframe.in_reply_to_user_id_str) is False:
        return dataframe.in_reply_to_user_id_str
        
    
def get_influencing_tweets_id(dataframe):
    if pd.isnull(dataframe.retweeted_status) is False:
        return dataframe.retweeted_status['id_str']
    elif pd.isnull(dataframe.quoted_status) is False:
        return dataframe.quoted_status['id_str']
    elif pd.isnull(dataframe.in_reply_to_user_id_str) is False:
        return dataframe.in_reply_to_status_id_str
        
    
def get_user_actions(dataframe):
    actions = []
    if pd.isnull(dataframe.retweeted_status) is False:
        actions.append('retweet')
    if pd.isnull(dataframe.quoted_status) is False:
        actions.append('quoted')
    if pd.isnull(dataframe.in_reply_to_user_id_str) is False:
        actions.append('replied')

    return actions


def keyword_extraction(claims):
    """
    Getting all keywords from a certain topic
    
    Args:
        claims: The claim published on fact-check website
        
    Returns:
        text: The keywords combination that can be used in the search query
    """
    # Import non English words
    #words = set(nltk.corpus.words.words())
    
    # Remove non-English words from the claim   
    claims = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in (nltk.word_tokenize(claims))]
    claims = ' '.join([s for s in claims if s])
    stopwords_list.append('us')
    stopwords_list.append('st')
    r = Rake(stopwords=stopwords_list) # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(claims)
    temp = r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.
    
    if len(temp[1:]) > 1:
        # Adding OR operation
        search_rule = " OR ".join(str(x) for x in temp[1:])

        search_rule = "({})".format(search_rule)

        # Adding language constraint
        search_rule = search_rule + ' lang:en'

        # Adding search keywords
        text = temp[0] + ' ' + search_rule
    else:
        text = temp[0] + ' lang:en'

    return text


def get_claim_dict(dataframe):
    """
    Getting all keywords from a certain topic
    
    Args:
        url: The claim from a Snopes.com fact-check website
        
    Returns:
        The keywords combination that can be used in the search query
    """
    topics = list(set(list(dataframe.topics)))
    
    for i in range(len(topics)):
        try:
            claims = get_claim(topics[i])
        except IndexError:
            claims = None
        except HTTPError:
            claims = None
            
        topics[i] = {topics[i]: claims}
        print("{}, Done".format(i))
    return topics


def drop_no_claim_rows(dataframe, topics):
    none_list = []; none_list_index = []
    drop_list = []
    
    topics_all = list(dataframe.topics)
        
    for i in range(len(topics)):
        if (list(topics[i].values())[0]) is None:
            none_list.append(list(topics[i].keys())[0])
            none_list_index.append(i)
            
    for i in range(len(topics_all)):
        if topics_all[i] in none_list:
            drop_list.append(i)

    # Drop non-useful rows if the claim is empty
    dataframe = dataframe.drop(dataframe.index[drop_list])
    cleaned_topic = [v for i,v in enumerate(topics) if i not in none_list_index]
    
    return dataframe, cleaned_topic


def list_to_txt(my_list, file_name):
    """
    Write a list of items into a txt file line by line
    
    Args:
        my_list: The list of items
        file_name: the file path and file name
    """
    with open('{0}.txt'.format(file_name), 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


def dict_to_csv(my_dict, file_name):
    """
    Write a dict into a csv file

    Args:
        my_dict: The dict object
        file_name: the file path and file name
    """
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in my_dict.items():
            writer.writerow([key, value])


def list_of_dicts_integration(list_of_dicts):
    return {k: v for d in list_of_dicts for k, v in d.items()}
    
    
def write_into_json(FILEPATH, obj):
    with open(FILEPATH+'.json', 'a', encoding='utf8') as file:
        for i in obj:
            json.dump(i, file)
            file.write('\n')        
    

def read_into_list(file_name):
    lines = []
    with open('true_tweets_snopes/{}.txt'.format(file_name)) as file:
        for line in file:  
            line = line.strip()
            lines.append(line)
    return lines

            
def check_percentage(file_name, temp):
    """Return the downloaded data's quality"""
    try:
        data_1 = load_json(file_name)
    
        aaa = list(data_1.id_str)
        if aaa: 
            counter = 0
            for i in aaa:
                if i in temp:
                    counter+=1
            print("Total tweets: {0} \n".format(len(aaa)))
            print("Snopes-related tweets: {0} \n".format(len(temp)))
            print("Matched tweets: {0} \n".format(counter))
            #print("This file contains {0}% of true information".format(counter/len(aaa)*100))
            print('----------------------------------------------------------\n')
        else:
            print("Total tweets: {0} \n".format(len(aaa)))
            print("Snopes-related tweets: {0} \n".format(len(temp)))
            #print("This file contains 0.0% of true information \n")
            print('------------------------------------------\n')
    except AttributeError as e:
            print('The file is empty \n'.format(e))
            print('----------------------------------------------------------\n')



def file_len(file_name):
    """Return the number of items in a line-separable file"""
    if os.stat(file_name).st_size == 0:
        return 0
    else:
        with open(file_name) as f:
            for i, l in enumerate(f):
                pass
        return i + 1


def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end


def number_of_tweets_in_range(start, end, time_list):
    counter = 0
    for i in time_list:
        if time_in_range(start, end, i):
            counter += 1
    #print(counter)
    return counter
    

def number_of_tweets_per_day(time_list):
    """Return datetime format object matching Twitter tweets created_at"""
    start = min(time_list).replace(hour=0, minute=0, second=0)
    end = max(time_list).replace(hour=0, minute=0, second=0)
    
    seq = []; curr = start
    seq.append(curr)
    for i in range((end - start).days + 1):
        curr = curr + timedelta(days = 1)
        seq.append(curr)
    
    tweets_per_day = []
    for i in range(len(seq)-1):
        tweets_per_day.append(number_of_tweets_in_range(seq[i], seq[i+1], time_list))
    
    del seq[-1]
    seq = [i.strftime("%m-%d-%Y") for i in seq]
    return seq, tweets_per_day


def parse_datetime(value):
    """Return datetime format object matching Twitter tweets created_at"""
    time_tuple = parsedate_tz(value)
    timestamp = mktime_tz(time_tuple)

    return datetime.fromtimestamp(timestamp)


def compare_datetime(file_name, snopes_time):
    """Return counter of tweets before snopes time and after snopes time"""
    counter_before, counter_after = 0, 0
    
    if os.stat(file_name).st_size != 0:
        created_time = list(load_json(file_name).created_at)
    
        for i in created_time:
            time = parse_datetime(i)
            if time <= snopes_time:
                counter_before += 1
            else:
                counter_after += 1

    #if (counter_before > counter_after):
    #print("Before: {0}, After, {1}\n".format(counter_before, counter_after))
    return counter_before, counter_after


def draw_time_series_barchart(file_name, snopes_time):
    created_time = [parse_datetime(i) for i in list(load_json(file_name).created_at)]
    
    x_axis, y_axis = number_of_tweets_per_day(created_time)
    snopes_invloving_time = snopes_time.strftime("%m-%d")

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    ax.bar(x_axis, y_axis)
    ax.axvline(x=snopes_invloving_time, c='r', label='Debunking time') 
    #ax.axes.get_xaxis().set_ticks([])
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim([0, max(y_axis)+30])
    plt.xlabel("Time Range From begining to End", fontsize=25)
    plt.ylabel("The number of involving tweets", fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.suptitle(file_name, fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, axis='y')
    plt.savefig('paper/figures/figure_5A.png', dpi=800)
    #plt.savefig('charts_without_snopes/{}.jpg'.format(file_name.split('.')[0].split('/')[1]))
    plt.show()
    
    
def split_list_to_chucks(arr, size):
    L = len(arr)
    assert 0 < size <= L
    s, r = divmod(L, size)
    t = s + 1
    a = ([arr[p:p+t] for p in range(0, r*t, t)] + [arr[p:p+s] for p in range(r*t, L, s)])
    return a


def divide_time_intervals(file_name, snopes_time):
    created_time = [parse_datetime(i) for i in list(load_json(file_name).created_at)]
    
    interval1, interval2, interval3, interval4, interval5 = [], [], [], [], []

    for i in created_time:
        if i < snopes_time:
            interval1.append(i)
        elif time_in_range(snopes_time, snopes_time + timedelta(days=1), i):
            interval2.append(i)
        else:
            interval3.append(i)

    interval1 = sorted(interval1, reverse=False)
    interval2 = sorted(interval2, reverse=False)
    interval3 = sorted(interval3, reverse=False)
    
    new_list = split_list_to_chucks(interval3, 3)
    
    interval3 = new_list[0]; interval4 = new_list[1]; interval5 = new_list[2]
    
    intervals = [interval1, interval2, interval3, interval4, interval5]
    
    for i in range(len(intervals) - 1):
        if len(intervals[i]) < 10:
            temp = 10-len(intervals[i])
            intervals[i] = intervals[i] + intervals[i + 1][:temp]
            intervals[i+1] = intervals[i+1][temp:]
    
    return intervals

            
def remove_mentions_urls(raw_text):
    text = re.sub(r'http\S+', '', raw_text)
    text = re.sub(r'@\w+ ?', '', text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'[^\x00-\x7F]+',' ',text)
    text = text.replace('RT', '')
    text = re.sub(r'[\n\r]+', '', text)
    return text


def split_dataset(file_name, intervals, index):
    testing = load_json(file_name)
    
    temp = [[], [], [], [], []]
    
    for i in range(len(testing)):
        if parse_datetime(testing.iloc[i].created_at) in intervals[0]:
            temp[0].append(i)
        elif parse_datetime(testing.iloc[i].created_at) in intervals[1]:
            temp[1].append(i)
        elif parse_datetime(testing.iloc[i].created_at) in intervals[2]:
            temp[2].append(i)
        elif parse_datetime(testing.iloc[i].created_at) in intervals[3]:
            temp[3].append(i)
        elif parse_datetime(testing.iloc[i].created_at) in intervals[4]:
            temp[4].append(i)
                    
    for i in range(len(temp)):
        test = testing.iloc[temp[i]]
        test['created_at'] = pd.to_datetime(test.created_at)
        test = test.sort_values(by='created_at')
        test['created_at'] = test['created_at'].astype(str)
        test.to_json('dataset_1/{0}/{1}.json'.format(index, i), orient='records', lines=True)            


def date_range(date_list, N):
    date = []
    if type(date_list[0]) != datetime:
        start = datetime.strptime(date_list[0],"%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(date_list[-1],"%Y-%m-%d %H:%M:%S")
    else:
        start = min(date_list)
        end = max(date_list)
        
    diff = (end - start)/N
    
    for i in range(N):
        date.append(start + diff * i)
    
    date.append(end)
    return list(date)
    

def get_training_vector_content(folder_name):
    dataset, text_list, hashtags, sentiment = [], [], [], []
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', token_pattern='[A-Za-z][\w\-]*', max_df=0.25)
    
    for i in range(5):
        dataset.append(load_json(folder_name + '{}.json'.format(i)))
    
    for i in range(len(dataset)):
        temp = [datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in list(dataset[i].created_at)]
        temp_text = [remove_mentions_urls(i) for i in list(dataset[i].text)]
        temp_hashtags = [i for i in list(dataset[i].entities)]
        temp_hashtag = []
        for i in temp_hashtags:
            if i['hashtags']:
                temp_hashtag.append(1)
            else:
                temp_hashtag.append(0)
                
        temp_sentiment = sentiment_score(temp_text)[3]
        
        time_interval = list(date_range(temp, 10))
        hashtag_element = []; sentiment_element = []
        
        for i in range(len(time_interval) - 1):
            temp_temp_hashtag, temp_temp_sentiment = [], []
            for j, x in enumerate(temp):
                if time_in_range(time_interval[i], time_interval[i+1], x):
                    temp_temp_hashtag.append(temp_hashtag[j])
                    temp_temp_sentiment.append(temp_sentiment[j])
                    
            hashtag_element.append(sum(temp_temp_hashtag))
            sentiment_element.append(np.mean(temp_temp_sentiment))
        
        temp_text_prime = vectorizer.fit_transform(temp_text)
        temp_text_prime = [np.argsort(temp_text_prime[i: i+100, :].toarray(), axis=1)[:, :20]
          for i in range(0, temp_text_prime.shape[0], 100)]
        
        text_list.append(temp_text_prime[0])
        hashtags.append(hashtag_element)
        sentiment.append(pd.Series(sentiment_element, dtype=object).fillna(0).tolist())
                
    return text_list, hashtags, sentiment


def get_training_vector_temp(folder_name):
    dataset, date_list, user_list, tweet_list, time_list = [], [], [], [], []
    
    for i in range(5):
        dataset.append(load_json(folder_name + '{}.json'.format(i)))
    
    for i in range(len(dataset)):
        temp = [datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in list(dataset[i].created_at)]
        temp_user = [i['id_str'] for i in list(dataset[i].user)]
        temp_tweet = list(dataset[i].id_str)
        
        time_interval = list(date_range(temp, 10))
        
        date_list_element, date_list_element_user, date_list_element_tweet = [], [], []
        
        for i in range(len(time_interval) - 1):
            temp_temp, temp_temp_user, temp_temp_tweet = [], [], []
            for j, x in enumerate(temp):
                if time_in_range(time_interval[i], time_interval[i+1], x):
                    temp_temp.append(x)
                    temp_temp_user.append(temp_user[j])
                    temp_temp_tweet.append(temp_tweet[j])
                    
            date_list_element.append(temp_temp)
            date_list_element_user.append(temp_temp_user)
            date_list_element_tweet.append(temp_temp_tweet)
            
        date_list.append(date_list_element)
        user_list.append(date_list_element_user)
        tweet_list.append(date_list_element_tweet)
        time_list.append(element_wise_subtraction(date_list_element))
        
    return date_list, user_list, tweet_list, time_list


def get_training_vector_user(folder_name):
    dataset, friends_list, verified_list, likes_list = [], [], [], []
    
    for i in range(5):
        dataset.append(load_json(folder_name + '{}.json'.format(i)))
    
    for i in range(len(dataset)):
        temp = [datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in list(dataset[i].created_at)]
        users_feature = [i for i in list(dataset[i].user)]
        
        temp_friends = [i['friends_count'] for i in users_feature]
        temp_verified = [int(i['verified']) for i in users_feature]
        temp_likes = list(dataset[i].favorite_count)
        
        time_interval = list(date_range(temp, 10))
        
        date_list_element_friend, date_list_element_verified, date_list_element_likes = [], [], []
        
        for i in range(len(time_interval) - 1):
            temp_temp_friends, temp_temp_verified, temp_temp_likes = [], [], []
            for j, x in enumerate(temp):
                if time_in_range(time_interval[i], time_interval[i+1], x):
                    temp_temp_friends.append(temp_friends[j])
                    temp_temp_verified.append(temp_verified[j])
                    temp_temp_likes.append(temp_likes[j])
                    
            date_list_element_friend.append(temp_temp_friends)
            date_list_element_verified.append(temp_temp_verified)
            date_list_element_likes.append(temp_temp_likes)
            
        friends_list.append(date_list_element_friend)
        verified_list.append(date_list_element_verified)
        likes_list.append(date_list_element_likes)
        
    return friends_list, verified_list, likes_list



def get_training_vector_structure(folder_name):
    dataset, users_list, followers_list, virality_list = [], [], [], []
    
    for i in range(5):
        dataset.append(load_json(folder_name + '{}.json'.format(i)))
    
    for i in range(len(dataset)):
        temp = [datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in list(dataset[i].created_at)]
        
        time_interval = list(date_range(temp, 10))
        
        for j in range(len(time_interval) - 1):
            dataframes = []
            users_list_tt, followers_list_tt, virality_list_tt = [], [], []
            for k, x in enumerate(temp):
                if time_in_range(time_interval[j], time_interval[j+1], x):
                    print(x, i, k)
                    dataframes.append(dataset[i].iloc[k])

            dataframes = pd.DataFrame(dataframes)
            number_cascades = number_of_cascades(dataframes, [])
            
            
            flatten_list = [item for sublist in number_cascades for item in sublist]
            users_list_tt.append(len(flatten_list))
        
            counter = 0
            for i in flatten_list:
                if i in [x['id'] for i,x in enumerate(list(dataframes.user))]:
                    counter += list(dataframes.user)[i]['friends_count']
            followers_list_tt.append(counter)
            d = str(number_cascades).count(",")+1
            if d-1 < 1:
                virality_list_tt.append(0)
            else:
                virality_list_tt.append(1/(d*(d-1)) * max([len(i) for i in number_cascades]))
    
        users_list.append(users_list_tt)
        followers_list.append(followers_list_tt)
        virality_list.append(virality_list_tt)
        
    return users_list, followers_list, virality_list


def element_wise_subtraction(lists):
    return_list = []
    
    for alist in lists:
        ttt = []
        if not alist:
            return_list.append(0)
        else:
            for j in range(len(alist)-1):
                ttt.append((alist[j+1] - alist[j])/timedelta(days=1))
                
            return_list.append(np.mean(ttt))
            
    return return_list
    

def number_of_retweets(dataset):
    counter = 0
    #users = list(set([dataset.iloc[i].user['id_str'] for i in range(len(dataset))]))
    for i in range(len(dataset)):
        if dataset.iloc[i].retweeted_status:
            counter += 1
        elif dataset.iloc[i].quoted_status:
            counter += 1
        elif dataset.iloc[i].in_reply_to_status_id_str:
            counter += 1
    
    users = list(set([dataset.iloc[i].user['id_str'] for i in range(len(dataset))]))
    
    
    return counter, users


def sentiment_score(sentence_list):
    sid = SentimentIntensityAnalyzer()

    neg, neu, pos, compound = [], [], [], []

    for sentence in sentence_list:
        ss = sid.polarity_scores(sentence)
        s = list(ss.values())
        neg.append(s[0])
        neu.append(s[1])
        pos.append(s[2])
        compound.append(s[3])
        
    return neg, neu, pos, compound


def tweet_text_analysis(filename, snope_time):
    dataset = load_json(filename)
    time = [parse_datetime(i) for i in list(dataset.created_at)]
    tweets = list(dataset.text)
    
    time = time[::-1]; tweets = tweets[::-1]
    
    vector = split_list_to_chucks(tweets, 30)    
    time_vector = split_list_to_chucks(time, 30) 
    
    for i in range(len(vector)):
        vector[i] = [remove_mentions_urls(j) for j in vector[i]]
    
    for i, x in enumerate(time_vector):
        if time_in_range(x[0], x[-1], snope_time):
            snopes_time_index = i
    
    return vector, snopes_time_index
        

def number_of_cascades(dataset, cascades_list):
    #cascades_list = []
    for i in range(len(dataset)):
        child = dataset.iloc[i].id_str
        if dataset.iloc[i].retweeted_status and type(dataset.iloc[i].retweeted_status) != float:
            parent = dataset.iloc[i].retweeted_status['id_str']
            if not any(parent in sl for sl in cascades_list):
                cascades_list.append([parent, child])
            else:
                for j, x in enumerate(cascades_list):
                    if parent in x:
                        cascades_list[j].append(child)
        elif hasattr(dataset.iloc[i], 'quoted_status') and dataset.iloc[i].quoted_status and type(dataset.iloc[i].quoted_status) != float:
            #if hasattr(dataset.iloc[i], 'quoted_status'):
            parent = dataset.iloc[i].quoted_status['id_str']
            if not any(parent in sl for sl in cascades_list):
                cascades_list.append([parent, child])
            else:
                for j, x in enumerate(cascades_list):
                    if parent in x:
                        cascades_list[j].append(child)
        elif dataset.iloc[i].in_reply_to_status_id_str:
            parent = dataset.iloc[i].in_reply_to_status_id_str
            if not any(parent in sl for sl in cascades_list):
                cascades_list.append([parent, child])
            else:
                for j, x in enumerate(cascades_list):
                    if parent in x:
                        cascades_list[j].append(child)
        else:
            cascades_list.append([dataset.iloc[i].id_str])
        #print('Done, {}'.format(i))
    return cascades_list
    
    
def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


def resort_dataframe(dataset):
    time_list = [parse_datetime(i) for i in list(dataset.created_at)]
    sorted_index = sorted(range(len(time_list)),key=time_list.__getitem__, reverse=False)
    
    dataset = dataset.reindex(sorted_index)
    
    return dataset


def sentiment_figure(filename, snope_time):
    vector, snopes_time_index = tweet_text_analysis(filename, snope_time)
    neg, neu, pos, compound = [], [], [], []
    
    for i in range(len(vector)):
        temp_A, temp_B, temp_C, temp_D = sentiment_score(vector[i])
        neg.append(s.mean(temp_A)) 
        neu.append(s.mean(temp_B))
        pos.append(s.mean(temp_C))
        compound.append(s.mean(temp_D))
    
    
    plt.plot(list(range(len(neg))), neg, 'g:', linewidth=2.4, label='neg')
    plt.plot(list(range(len(neu))), neu, 'm-.', label='neu')
    plt.plot(list(range(len(pos))), pos, 'b:', linewidth=2.4, label='pos')
    plt.axvline(x=11, c='r', linewidth=3, label='Snopes time') 
    plt.legend(loc=6, fontsize=11)
    plt.xlabel('Time Interval', fontsize=15)
    plt.ylabel('Average Sentiment Score', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('paper/figures/test1.png', dpi=600)
    plt.show()
      
    plt.plot(list(range(len(compound))), compound, 'g-.', label='compound')
    plt.axis([None, None, -1, 1])
    plt.axvline(x=11, c='r', linewidth=3, label='Snopes time') 
    plt.axhline(y=0, c='black') 
    plt.legend(loc='upper left', fontsize=11)
    plt.xlabel('Time Interval', fontsize=15)
    plt.ylabel('Average Sentiment Score', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('paper/figures/test.png', dpi=600)
    plt.show()

