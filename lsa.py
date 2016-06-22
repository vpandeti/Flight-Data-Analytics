from __future__ import division
import os
import csv
import sys

import mysql.connector
import pandas
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from flask import Flask, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from flask import request

project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')
app = Flask(__name__, template_folder=template_path, static_folder=static_path)
wsgi_app = app.wsgi_app
n_components = 54
n_features = 1000
lsa_clusters = 1

# connect
db = mysql.connector.connect(host='localhost',database='visualization',user='root',password='pass4root')

def getMinMaxDateForChannelVideos(channel_title):
    try:
        s  = "SELECT MIN(DATE(PUBLISHED_AT)) as MINDATE, MAX(DATE(PUBLISHED_AT)) as MAXDATE FROM VIDEO WHERE CHANNEL_TITLE ='" + channel_title + "'";
        cursor = db.cursor()

        # execute SQL select statement
        cursor.execute(s)

        # get the number of rows in the resultset
        row = cursor.fetchone()

        # get and display one row at a time.
        while row is not None:
            return row[0], row[1]
    except:
        e = sys.exc_info()[0]
        print e

def getVideosUploadedOverTime(channel_title, category, aggregate_for_category):
    min_date = ''
    max_date = ''
    min_date, max_date = getMinMaxDateForChannelVideos(channel_title)
    s = ('select video_title, date(published_at) as publishedat,\'' + min_date + '\' as channeldate,  DATEDIFF(date(published_at),\'' + min_date + '\') as daysFromStart ' \
        'from video where channel_title  = \'' + channel_title + '\' order by publishedat')

    cursor = db.cursor()

    # execute SQL select statement
    cursor.execute(s)

    # get the number of rows in the resultset
    row = cursor.fetchone()

    # get and display one row at a time.
    while row is not None:
        return row[0], row[1]
        row = cursor.fetchone()

def getChannelTrend():
    cursor = db.cursor()

    # execute SQL select statement
    cursor.execute("SELECT * FROM LOCATION")

    # commit your changes
    db.commit()

    # get the number of rows in the resultset
    numrows = int(cursor.rowcount)

    # get and display one row at a time.
    for x in range(0,numrows):
        row = cursor.fetchone()
        print row[0], "-->", row[1]



@app.route('/')
def d3():
    return render_template('visualization.html')

@app.route('/channel', methods=['GET','POST'])
def getChannelStatistics():
    viewId= request.args.get('view')
    sourceId = request.args.get('source')
    categoryId = request.args.get('category')
    if "V1" == viewId:
        getVideosUploadedOverTime(sourceId, categoryId, False)

@app.route('/lsa')
def lsa():
    global n_components
    global lsa_clusters
    global n_features
    svd = TruncatedSVD(n_components)
    svd_normalizer = Normalizer(copy=False)
    svd_lsa = make_pipeline(svd, svd_normalizer)
    '''data_categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]'''
    channel_tags = {}
    category_tags = {}
    file_obj = open('tags.csv','r')
    csv_data = csv.DictReader(file_obj, delimiter=',')
    for row in csv_data:
        if row["channel_title"] not in channel_tags:
            channel_tags[row["channel_title"]] = []
        if row["category_name"] not in category_tags:
            category_tags[row["category_name"]] = []
        channel_tags[row["channel_title"]].append(unicode(row["tags"], errors='replace').replace('|', ' '))
        category_tags[row["category_name"]].append(unicode(row["tags"], errors='replace').replace('|', ' '))
    for key in channel_tags:
        print channel_tags[key]
    for key in category_tags:
        print category_tags[key]
    channel_key = 'Automobile Magazine'
    # data = fetch_20newsgroups(subset='all', categories=data_categories, shuffle=True, random_state=42)
    # data = channel_tags[channel_key]
    svd_vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2, stop_words='english', use_idf=True)
    '''svd_vectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2),
                      use_idf=1,smooth_idf=1,sublinear_tf=1)'''
    category_list = {}
    for key in channel_tags:
        data = channel_tags[key]
        X = svd_vectorizer.fit_transform(data)
        X = svd_lsa.fit_transform(X)
        doc_explained_variance = svd.explained_variance_ratio_.sum()
        kmeans = KMeans(n_clusters=lsa_clusters, init='k-means++', max_iter=100, n_init=1)
        kmeans.fit(X)
        doc_original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_)
        doc_order_centroids = doc_original_space_centroids.argsort()[:, ::-1]

        lsa_data = []
        terms = svd_vectorizer.get_feature_names()

        '''vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
        dtm = vectorizer.fit_transform(data)
        word_dict = {}
        array = dtm.toarray()
        dataframe = pandas.DataFrame(dtm.toarray(),index=data,columns=vectorizer.get_feature_names()).head(10)'''
        data = []
        for i in range(lsa_clusters):
            data = []
            for ind in doc_order_centroids[i, :10]:
                print terms[ind]
                data.append(terms[ind])
            #lsa_data.append(data)
        category_list[key] = data
    f = open('data1.json','w')
    json_data = pandas.json.dumps(category_list)
    print json_data
    f.write(json_data)
    f.close()
    return json_data

if __name__ == "__main__":
    #app.run('localhost', '5555')
    lsa()