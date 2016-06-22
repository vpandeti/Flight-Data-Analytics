from __future__ import division
import random
import sys
import os
import numpy as np
import pandas
from sklearn.decomposition import TruncatedSVD
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from flask import Flask, render_template
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')
app = Flask(__name__, template_folder=template_path, static_folder=static_path)
wsgi_app = app.wsgi_app
labels = []
random_samples = []
adaptive_samples = []
data_file = pandas.read_csv('flights1987.csv', low_memory=False)
data_file = data_file.fillna(0)
del data_file['UniqueCarrier']
del data_file['TailNum']
del data_file['AirTime']
del data_file['Origin']
del data_file['Dest']
del data_file['TaxiIn']
del data_file['TaxiOut']
del data_file['Cancelled']
del data_file['CancellationCode']
del data_file['Diverted']
del data_file['CarrierDelay']
del data_file['WeatherDelay']
del data_file['NASDelay']
del data_file['SecurityDelay']
del data_file['LateAircraftDelay']
samplesize = 200
n_components = 200
n_features = 1000
lsa_clusters = 4

@app.route('/')
def d3():
    return render_template('d3.html')

def clustering():
    global data
    global data_file
    features = data_file[['DepTime', 'CRSDepTime', 'ArrTime']]
    k = 3
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    kmeans_centres = kmeans.cluster_centers_
    labels = kmeans.labels_
    data_file['kcluster'] = pandas.Series(labels)

def random_sampling():
    # Random samples
    global data
    global data_file
    global random_samples
    global samplesize
    features = data_file[['DepTime', 'CRSDepTime', 'ArrTime']]
    data = np.array(features)
    rnd = random.sample(range(len(data_file)), samplesize)
    for j in rnd:
        random_samples.append(data[j])

def adaptive_sampling():
    # Adaptive samples
    global data_file
    global adaptive_samples
    # features = data_file[['DepTime', 'CRSDepTime', 'ArrTime']]
    size_sample = samplesize

    kcluster0 = data_file[data_file['kcluster'] == 0]
    kcluster1 = data_file[data_file['kcluster'] == 1]
    kcluster2 = data_file[data_file['kcluster'] == 2]

    size_kcluster0 = len(kcluster0) * size_sample / len(data_file)
    size_kcluster1 = len(kcluster1) * size_sample / len(data_file)
    size_kcluster2 = len(kcluster2) * size_sample / len(data_file)

    sample_cluster0 = kcluster0.ix[random.sample(kcluster0.index, int(size_kcluster0))]
    sample_cluster1 = kcluster1.ix[random.sample(kcluster1.index, int(size_kcluster1))]
    sample_cluster2 = kcluster2.ix[random.sample(kcluster2.index, int(size_kcluster2))]

    adaptive_samples = pandas.concat([sample_cluster0, sample_cluster1, sample_cluster2])

clustering()
random_sampling()
adaptive_sampling()

@app.route('/pca_random')
def pca_random():
    """PCA Analysis"""
    data_columns = []
    try:
        global random_samples
        pca_data = PCA(n_components=2)
        X = random_samples
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
        pca_variance = pca_data.explained_variance_ratio_
        data_columns['variance'] = pandas.DataFrame(pca_variance)[0]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/pca_adaptive')
def pca_adaptive():
    """PCA Analysis"""
    data_columns = []
    try:
        global adaptive_samples
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        pca_data = PCA(n_components=2)
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
        pca_variance = pca_data.explained_variance_ratio_
        data_columns['variance'] = pandas.DataFrame(pca_variance)[0]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/isomap_random')
def isomap_random():
    # Isomap
    data_columns = []
    try:
        global random_samples
        isomap_data = manifold.Isomap(n_components=2)
        X = isomap_data.fit_transform(random_samples)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/isomap_adaptive')
def isomap_adaptive():
    # Isomap
    data_columns = []
    try:
        global adaptive_samples
        isomap_data = manifold.Isomap(n_components=2)
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        isomap_data.fit(X)
        X = isomap_data.transform(X)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/mds_euclidean_random')
def mds_euclidean_random():
    # MDS
    data_columns = []
    try:
        global random_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/mds_euclidean_adaptive')
def mds_euclidean_adaptive():
    data_columns = []
    try:
        global adaptive_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        similarity = pairwise_distances(X, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/mds_cosine_random')
def mds_cosine_random():
    data_columns = []
    try:
        global random_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='cosine')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/mds_cosine_adaptive')
def mds_cosine_adaptive():
    data_columns = []
    try:
        global adaptive_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        similarity = pairwise_distances(X, metric='cosine')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/mds_correlation_random')
def mds_correlation_random():
    data_columns = []
    try:
        global random_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='correlation')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/mds_correlation_adaptive')
def mds_correlation_adaptive():
    data_columns = []
    try:
        global adaptive_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        similarity = pairwise_distances(X, metric='correlation')
        X = mds_data.fit_transform(similarity)
        data_columns = pandas.DataFrame(X)
        data_columns['departure'] = data_file['DepTime'][:samplesize]
        data_columns['arrival'] = data_file['ArrTime'][:samplesize]
        data_columns['clusterid'] = data_file['kcluster'][:samplesize]
    except:
        e = sys.exc_info()[0]
        print e
    return pandas.json.dumps(data_columns)

@app.route('/lsa')
def lsa():
    global n_components
    global lsa_clusters
    global n_features
    svd = TruncatedSVD(n_components)
    svd_normalizer = Normalizer(copy=False)
    svd_lsa = make_pipeline(svd, svd_normalizer)
    data_categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    data = fetch_20newsgroups(subset='all', categories=data_categories, shuffle=True, random_state=42)
    data_labels = data.target
    svd_vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2, stop_words='english', use_idf=True)
    X = svd_vectorizer.fit_transform(data.data)
    X = svd_lsa.fit_transform(X)
    doc_explained_variance = svd.explained_variance_ratio_.sum()
    kmeans = KMeans(n_clusters=lsa_clusters, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(X)
    doc_original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_)
    doc_order_centroids = doc_original_space_centroids.argsort()[:, ::-1]

    lsa_data = []
    terms = svd_vectorizer.get_feature_names()
    for i in range(lsa_clusters):
        data = []
        for ind in doc_order_centroids[i, :10]:
            print terms[ind]
            data.append(terms[ind])
        lsa_data.append(data)
    return pandas.json.dumps(lsa_data)

if __name__ == "__main__":
    app.run('localhost', '5555')