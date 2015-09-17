
# coding: utf-8

# In[13]:

# Python Twitter API 
# Authored By Elliott Miller - DataFighter
# twitter handle @datafighter1
# email: ellmill00@gmail.com

# I just like using pandas to parse and write things
# This 
import pandas as pd

# Will also use the json library. 
import json

# python-twitter library
# you can obtain this by using pip python-twitter
# documentation at https://pypi.python.org/pypi/python-twitter/
import twitter
from twitter import *

# library that makes requests easy to do
import requests

# So I can deal with rate limiting
import time

config = {
        "consumer_key":'8znuRJV8XI4o10zaGs2BY0kd4',
        "consumer_secret":'c4UxsLgqqbFLZj7kLDpQunv9wSGnd8mmRxHuUCqcmH8VsAqb0Y',
        "access_token_key":'3439153456-fLRa1GuO7pMJOfl92a7MOfx1Zkbw3Xl478dXfNr',
        "access_token_secret":'5psGt3VrNtULGmGZ6SucFFt8ewSo56eL5HcddYImLAh6V'}

# Takes in a group name and outputs a .json file that has the same name as the input parameter
# USe of this function will also require an api key
def WriteMeetupEvents(GroupName):
    #Put in your API key here
    #You can get one from https://secure.meetup.com/meetup_api/key/
    api_key = "5db37233e5711225b62251680323943"
    
    #Get request for the events list of a group
    #This is simplified using the python requests library
    response = requests.get("https://api.meetup.com/2/events?key=5db37233e5711225b62251680323943&group_urlname="+GroupName+"&sign=true")
    
    #Write the server respons as a json
    #This is unparsed. 
    with open(GroupName+'.json', 'w') as fp:
        json.dump(response.json(), fp)
        
def ManageTwitterRateLimit(api,data_requirement):
    '''
    Designed to call the requested function and ensure the rate limit is not exceeded.

    :param api_call:
    :param query:
    :param data_limit:
    :return:
    '''

    data = []
    iterations = 0
    data_gathered = 0
    max_iterations = 10000
    while ((data_gathered < data_requirement) & (iterations <= max_iterations)):

        remaining = api.remaining()
        if remaining!='failed':
            result = api.call()
            # Twitter 429 error response: { "errors": [ { "code": 88, "message": "Rate limit exceeded" } ] }
            if result!='failed':
                data += result
                data_gathered = len(data)
                print "Amount gathered: " + str(data_gathered)
            else: # 'errors' in result.keys():
                last_error_time = time.time()
                print "Oooopps, let's give it a minute."
                time.sleep(60) # Wait a minute, literally.

        else:
            time_now = time.time()
            last_error_time = time.time()
            print "It's been "+ str(time_now-last_error_time) +"Gimme 9."
            time.sleep(9) # Wait a minute, well 9s, literally.

        iterations += 1

    return data


def GetTwitterQuery(query,data_limit):

    class apiGetSearch():
        def __init__(self,query):

            self.api = twitter.Api(
                consumer_key = config['consumer_key'],
                consumer_secret = config['consumer_secret'],
                access_token_key = config['access_token_key'],
                access_token_secret = config['access_token_secret'])
            self.query = query

        def call(self):
            try:
                query_result = self.api.GetSearch(self.query)
            except TwitterError:
                query_result = "failed"
            return query_result

        def remaining(self):
            try:
                rate_status = self.api.GetRateLimitStatus()
                queries_remaining = rate_status['resources']['search']['/search/tweets']['remaining']
            except TwitterError:
                queries_remaining = "failed"
            return queries_remaining

    ags = apiGetSearch(query)
    statuses = ManageTwitterRateLimit(ags,data_limit)
    df = pd.DataFrame(statuses)
    return df

# This Function takes in a parameter as a screenname and then writes a json file
# With the screenname as the filename
def WriteTwitterStatuses(ScreenName):
    
    # Create the api
    # You need to input your own twitter keys and tokens
    # You can get the keys by registering at https://apps.twitter.com/
    api = twitter.Api(
        consumer_key = config['consumer_key'],
        consumer_secret = config['consumer_secret'],
        access_token_key = config['access_token_key'],
        access_token_secret = config['access_token_secret'])

    
    # Get all of the statuses. It Outputs to a list
    statuses = api.GetUserTimeline(screen_name = ScreenName)

    # ***The following two lines require pandas***    

    # make a pandas dataframe from the status array    
    df = pd.DataFrame(statuses)
    
    # write the twitter statuses as a .json file
    # using pandas    
    # The File Name is just the screenname
    df.to_json(str(ScreenName)+'.json')
    
    # Uncomment the next line to print statuses
    # print [s.text for s in statuses]
    
# Uncomment the next line and run to verify your credentials on Twitter:
# print api.VerifyCredentials()


# In[9]:




# In[11]:

# Takes in a group name and outputs a .json file that has the same name as the input parameter
# USe of this function will also require an api key
def WriteMeetupEvents(GroupName):
    #Put in your API key here
    #You can get one from https://secure.meetup.com/meetup_api/key/
    api_key = "5db37233e5711225b62251680323943"
    
    #Get request for the events list of a group
    #This is simplified using the python requests library
    response = requests.get("https://api.meetup.com/2/events?key=5db37233e5711225b62251680323943&group_urlname="+GroupName+"&sign=true")
    
    #Write the server respons as a json
    #This is unparsed. 
    with open(GroupName+'.json', 'w') as fp:
        json.dump(response.json(), fp)


# In[14]:

WriteMeetupEvents('Data-Community-DC')


# In[ ]:



