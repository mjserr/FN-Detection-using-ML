import csv, re
from xml.etree.ElementInclude import include
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import equal
import tweepy
from tweepy import tweet
from csv import reader, writer
from more_itertools import unique_everseen
import codecs

#Twitter API's
auth = tweepy.OAuthHandler("API KEY", "API KEY SECRET")
auth.set_access_token("ACCESS TOKEN", "ACCESS TOKEN SECRET")
api = tweepy.API(auth)

#################################################################################################################
#Function to scrape Tweets from all the Verified Twitter News Account
def twtScrape(accName):
    output=[]
    for i in range(len(accName)):
        givenName = accName[i]
        for tweets in tweepy.Cursor(api.user_timeline, id=givenName).items(10):
            tweetID = tweets.id
            status = api.get_status(tweetID, tweet_mode = 'extended')
            fullTwt = status.full_text  
            pushToCSV = {"title" : tweets.author.name, "text" : fullTwt, "label" : "REAL"}
            output.append(pushToCSV)
    #Inserting all the scraped Tweets inside in one CSV File
    df = pd.DataFrame(output)
    df.to_csv('realNews.csv', index = False)
    print("Scraping Successful") #End of Tweet Scraping
    return df

#################################################################################################################
#Funtion to merge the 'realNews.csv' and 'fakeNews.csv' in one CSV file 'dataset.csv'
def mergeCSV(csvFile):

    
    datas = pd.concat(map(pd.read_csv,['realNews.csv', 'fakeNews.csv']), ignore_index = False)
    datas.dropna()
    datas.drop_duplicates(subset="text", keep=False, inplace=True)
    datas.to_csv('dataset.csv', index=False)
    return datas

#################################################################################################################
#Function to run the Jupyter File to train the 'dataset.csv'
def trainModel():
    import asyncio
    import sys

    if sys.platform:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    #Training of 'dataset.csv' that will be use for the system
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    #Opening and Running the Jupyter File to start the training of the data
    filename = 'Fake_News_Detection.ipynb'
    with open(filename) as ff:
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
    
    ep = ExecutePreprocessor(timeout=400, kernel_name='')

    nb_out = ep.preprocess(nb_in)

# Function to get URL
def Find(string):
  
    # findall() has been used 
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)      
    return [x[0] for x in url]

# Function to convert  
def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 
    
#################################################################################################################
#Function to run the main system
def runSystem():
    app = Flask(__name__)
    tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    dataframe = pd.read_csv('dataset.csv')
    dataframe = dataframe.dropna()
    x = dataframe['text']
    y = dataframe['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    def fake_news_det(news):
        tfid_x_train = tfvect.fit_transform(x_train)
        tfid_x_test = tfvect.transform(x_test)
        input_data = [news]
        vectorized_input_data = tfvect.transform(input_data)
        prediction = loaded_model.predict(vectorized_input_data)
        return prediction

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/index')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            message = request.form['message'] 
            message.encode('unicode_escape')
            result_row= []
            author = ""
            msg = ""
            pattern = r"http\S+"
            tweetmsg = ""
            nlink = ""
            link = ""
            if len(message.split())<3:
                print(message)
                print("cannot predict")
                pred =['TRY']
                print(pred) 
                tweetmsg = "Try again with more than 2 words"
                author = "Unknown"
                link = "Unknown"
            else:
                with open('dataset.csv', encoding='utf-8') as csv_file: #insert the csv
                    csv_read = csv.reader(csv_file, delimiter=',')
                    for row in csv_read:
                        search_terms = [message]
                        if any([term in row[1] for term in search_terms]): #row[2], index of selected column
                            result_row.append(row)
                    if len(result_row) == 1:
                        print(result_row)
                        author = result_row[0][0]
                        msg = result_row[0][1]
                        tweetmsg = re.sub(pattern, "", msg)
                        nlink = listToString(Find(result_row[0][1]))
                        link = nlink.replace("[", "")
                        pred = fake_news_det(message)
                        if len(link)==0:
                            link = "None"
                        else:
                            link = link
                        print(author)     
                        print(tweetmsg)   
                        print(link)      
                    else:
                        print("This search is skipped because is has:", len(result_row), "values") 
                        pred = fake_news_det(message)
                        tweetmsg = message
                        author = "Unknown"
                        link = "Unknown"
                        print(author)
                        print(tweetmsg)
                        print(pred)
                        print(link)  
                        if pred != 'REAL':
                            fake = ["Unknown",  message, "FAKE"]
                            with open('fakeNews.csv', 'a') as fake_object:
                                fakeData = writer(fake_object)
                                fakeData.writerow(fake)
                        else:
                            pred =['FAKE']
                            print(pred)    
                            fake = ["Unknown",  message, "FAKE"]
                            with open('fakeNews.csv', 'a') as fake_object:
                                fakeData = writer(fake_object)
                                fakeData.writerow(fake)
                                        
            return render_template('results.html', prediction=pred,usertxtauthor=author,usertxt=tweetmsg,userlink=link)
        else:
            return render_template('results.html', prediction="Something went wrong")

    @app.route('/results', methods=['POST'])
    def results():
        return render_template('results.html')

    if __name__ == '__main__':
        app.run(debug=True)

#Username of Verified News Twitter Account

accName=['ABSCBNNews', '24OrasGMA', 'News5PH', 'TVPatrol', 'gmanetwork', 'gmapinoytv', 
'PTVph', 'inquirerdotnet', 'dost_pagasa', 'dzrhnews', 'rapplerdotcom', 'UnangHirit', 'gtvphilippines', 'TV5manila',
'ANCALERTS', 'PhilstarNews', 'KM_Jessica_Soho', 'gmanews', 'GMA_PA', 'manilabulletin', 'ABSCBN_Showbiz', 'IWitnessGMA',
'saksi', 'cnnphilippines', 'ABSCBN', 'phivolcs_dost', 'dzbb', 'gmanewsbreaking', 'stateofdnation', 'CLTV36']

mergeCSV(twtScrape(accName))
trainModel()
runSystem()