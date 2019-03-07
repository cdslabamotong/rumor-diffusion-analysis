# -*- coding: utf-8 -*-
# Author: Chen Ling

from searchtweets import gen_rule_payload, load_credentials, collect_results

premium_search_args = load_credentials("~/.twitter_keys.yaml",
                                       yaml_key='search_tweets_api',
                                       env_overwrite=False)

aaa = 'provide hackers (online accounts OR friend request OR stranger OR computer OR access OR accepting)-snopes lang:en'
# testing with a sandbox account
rule = gen_rule_payload(aaa,
                        from_date="2017-04-01",
                        to_date="2017-09-30",
                        results_per_call=500)


#count_rule = gen_rule_payload("hillary weapons isis lang:en has:mentions is:retweet",
#                              count_bucket="day")


tweets_fake = collect_results(rule, 
                              max_results=500, 
                              result_stream_args=premium_search_args)

#tweets_fake_count = collect_results(count_rule, max_results=500, result_stream_args=premium_search_args)

#[print(tweet.all_text, tweet.created_at_datetime, '\n') for tweet in tweets_fake[0:10]]

#[print(tweet, '\n') for tweet in tweets_fake[0:1]]

#count_rule = gen_rule_payload("hillary weapons isis lang:en has:mentions is:retweet has:links", count_bucket="day")

#counts = collect_results(rule, result_stream_args=premium_search_args)




