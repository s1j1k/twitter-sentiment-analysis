from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import re
import numpy as np
from scipy import stats
from sklearn.metrics import euclidean_distances

# helper function
def decode_str(str):
    # fix encoding for emojis
    soup = BeautifulSoup(str,"html5lib")
    text = soup.text.encode('latin1').decode('utf-8','ignore')
    return text

# preprocess the text data
#import emoji

replace_strings = {
    "_TWITTER-ENTITY_":"",
    "_TWITTER-ENTITY_:":"",
    "& amp ;":"and",
    "& lt ;":"<",
    "& gt ;":">"
}

def preprocess(tweet):
    """
    preprocess the text of the tweet
    """


    # replace the random strings
    for x in replace_strings:
        tweet = tweet.replace(x, replace_strings[x])

    # replace emojis with the meanings ->actually VADER can handle this!
    #tweet = emoji.demojize(tweet)

    # remove any unncessary punctuation
    # this will make emoji meanining with _ turn into words with spaces like :red_heart: becomes red heart
    # retain the emoticons
    #tweet = re.sub(r"[;:_]![D)()]+"," ",tweet,flags=re.MULTILINE)


    # convert all text lowercase
    #tweet = tweet.lower()

    # remove any urls
    tweet = re.sub(r"http\S+|www\S+|https\S+","",tweet,flags=re.MULTILINE)

    # remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#|\\',"",tweet)

    # ensure polarity of negations is captured

    # exception in english
    if "can't" in tweet.lower():
        tweet = tweet.replace("can't","can not")
        tweet = tweet.replace("CAN'T","CAN NOT")
        tweet = re.sub(r'[Cc][Aa][Nn]\'[tT]','can not',tweet)

    # exception in english
    if "shan't" in tweet.lower():
        tweet = tweet.replace("shan't","shall not")
        tweet = tweet.replace("SHAN'T","SHALL NOT")
        tweet = re.sub(r'[Ss][Hh][Aa][Nn]\'[tT]','shall not',tweet)

    # should cover the rest of the contractions
    if "n't" in tweet.lower():
        tweet = tweet.replace("n't"," not")
        tweet = tweet.replace("N'T"," NOT")
        tweet = re.sub(r'[Nn]\'[tT]',' not',tweet)

    
    # replace repeating letters more than 2 times 
    tweet = re.sub('^NO+', 'NO', tweet)
    tweet = re.sub('^no+', 'no', tweet)
    tweet = re.sub('^N[oO]+', 'No', tweet)
    tweet = re.sub('^n[oO]+', 'no', tweet)

    tweet = re.sub('([a-zA-Z])\\1+', '\\1\\1', tweet)
    tweet = re.sub('([a-zA-Z])\\1+', '\\1\\1', tweet)

    return tweet

def get_prediction(x):
            if x <= -0.05:
                return 'negative'
            if x >= 0.05:
                return 'positive'
            return -1
    
class UnsupervisedClassifier():
    """
    Unsupervised learner for assignment 3.
    """
    def __init__(self, n_clusters=2, mode=None):
        """
        n_clusters: number of clusters to be used in the KMeans clustering, int
        """
        self.n_clusters = n_clusters
        self.Mode = mode

    
    def fit(self, raw_text, emb_text):
        """
        raw_text: raw tweets used as training data, dtype pd.Series
        emb_text: tweets in a numeric formatting for later clustering if first method unsucessful
        """
        # prepare the text
        raw_text = raw_text.apply(decode_str)
        raw_text = raw_text.apply(preprocess)

        # perform sentiment analysis with VADER
        analyser = SentimentIntensityAnalyzer()
        scores = raw_text.apply(lambda x: analyser.polarity_scores(x)['compound'])

        predictions = scores.apply(get_prediction) 

        # train the clustering algorithm
        X_train = np.array(emb_text[predictions != -1].to_list())
        y_train = np.array(predictions[predictions != -1])

        # predict values for the rest of the X in the training set 
        X_pred = np.array(emb_text[predictions == -1].to_list())

        # get labels for the data which did not work well with VADER
        kmeans = KMeans(n_clusters=self.n_clusters).fit(X_train)
        clusters = kmeans.fit_predict(X_train) 

        # for each cluster get the most prevalent class
        labels = np.array([stats.mode(y_train[clusters == i])[0][0] for i in range(self.n_clusters)])

        y_pred = np.array(labels[kmeans.predict(X_pred)])

        self._X = np.concatenate([X_train, X_pred])
        self._y = np.concatenate([y_train, y_pred])

        self._kmeans = KMeans(n_clusters=self.n_clusters).fit(self._X)
        clusters = kmeans.fit_predict(self._X) 
        self._labels = np.array([stats.mode(self._y[clusters == i])[0][0] for i in range(self.n_clusters)])

    def predict(self, X_raw, X_emb):
        """
        X_raw: pd.Series
        X_emb: pd.Series
        """
        if self.mode == 'simple':
            closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self._y[closest]
        # try to predict with VADER
        X_raw = X_raw.apply(decode_str)
        X_raw = X_raw.apply(preprocess)

        # perform sentiment analysis with VADER
        analyser = SentimentIntensityAnalyzer()
        scores = X_raw.apply(lambda x: analyser.polarity_scores(x)['compound'])

        predictions = scores.apply(get_prediction)

        # use the already fitted cluster to predict labels where sentiment did not get predicted
        X_pred = np.array(X_emb[predictions == -1].to_list())
        y_pred = np.array(self._labels[self._kmeans.predict(X_pred)])

        # else predict with KNN
        predictions[predictions == -1] = y_pred

        return np.array(predictions.to_list())


