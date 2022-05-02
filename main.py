import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from sklearn.metrics import adjusted_rand_score


articles_df = pd.read_json('parsedText.json', lines=True)


content = articles_df['text']
ids = articles_df['id']
test = content.head()
words = []
for row in content:
    row.strip()
    words.append(row)

test_list = ' '.join(words)

toeknized = word_tokenize(test_list)
stop_words = list(stopwords.words("english"))
extra = ['says','say','said','one','would','also','get','may','two','like','could', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
for i in extra:
    stop_words.append(i)
filtered_list = []

for word in toeknized:
    if word.casefold() not in stop_words:
        filtered_list.append(word)

# for word in filtered_list:
#     if word.startswith('//'):
#         filtered_list.remove(word)


lemmatizer = WordNetLemmatizer()


final_tweets = [lemmatizer.lemmatize(word) for word in filtered_list]


frequency_distribution = FreqDist(final_tweets)

# plt.figure(figsize=(10, 10))
# frequency_distribution.plot(20)
# print(frequency_distribution.most_common(20))
# print(len(ids))

# print(articles_df.head())

##################################
# K-Means
##################################

import time
start_time = time.time()

documents = content

vectorizer = TfidfVectorizer(stop_words=extra)
X = vectorizer.fit_transform(documents)

true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
model.fit(X)


# print("Top Terms Per Cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
id_clus = []
words = []
for i in range(true_k):
    id_clus.append(i),
    for ind in order_centroids[i, :10]:
        words.append(terms[ind]),
    # print

labels = model.labels_
labels = labels.tolist()

# print(labels)


cats = []
for i in labels:
    if i==0:
        cats.append('community matters')
    elif i==1:
        cats.append('local politics')
    elif i==2:
        cats.append('healthcare')
    elif i==3:
        cats.append('intl politics')
    elif i==4:
        cats.append('school system')
    elif i==5:
        cats.append('crime')

articles_df['label'] = cats

corp = articles_df[['text', 'label']].copy()

corp.to_csv('corp.csv')

# print("\n")
# print("Prediction")
#
# Y = vectorizer.transform(["new school being built"])
# prediction = model.predict(Y)
# print(prediction)

# kmeans_kwargs = {"init": "k-means++", "n_init": 10,"max_iter": 300,"random_state": 42,}
#
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(X)
#     sse.append(kmeans.inertia_)
#
# kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
# kl.elbow


# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 21), sse)
# plt.xticks(range(1, 21))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.tight_layout()
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
