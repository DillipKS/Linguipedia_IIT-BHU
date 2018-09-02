import numpy as np
import pandas as pd

import re, nltk, time
import gensim, math
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn import linear_model, metrics

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
split_per = 0.75    # Training-Test split ratio


def preprocess(dataframe):
    '''
    Preprocess the tweets in train.csv and store it into a csv file with ID and Label
    '''
    columns=['id','label', 'tweet']    
    tweets = []
    for row in dataframe.itertuples(index=True, name='Pandas'):
        mystring = re.sub('[^A-Za-z]+', ' ', row[3])
        text = ' '.join([word.lower() for word in mystring.split() if word not in stop_words and len(word)>=2])
        tweets.append((row[1], row[2], text))
    
    tweets_df = pd.DataFrame.from_records(tweets, columns=columns)
    with open('tweets_clean.csv', 'w') as file:
        tweets_df.to_csv(file, index=False)
    
    return tweets_df


def test_preprocess(dataframe):
    '''
    Preprocess the tweets in test.csv and store it into a csv file with ID and Label
    '''
    columns=['id', 'tweet']
    tweets = []
    for row in dataframe.itertuples(index=True, name='Pandas'):
        mystring = re.sub('[^A-Za-z]+', ' ', row[2])
        text = ' '.join([word.lower() for word in mystring.split() if word not in stop_words and len(word)>=2])
        tweets.append((row[1], text))

    tweets_df = pd.DataFrame.from_records(tweets, columns=columns)
    with open('tweets_test.csv', 'w') as file:
        tweets_df.to_csv(file, index=False)
	
    return tweets_df


def test_data_prediction(X_new, ID_test):
    '''
    Predict the class labels for held out test data and print into a csv file
    '''
    X_new_vec = vectorizer.transform(X_new)
    pred = logreg.predict(X_new_vec)
    columns=['id', 'label']
    predictions = []
    for i in range(len(ID_test)):
        predictions.append((ID_test[i], pred[i]))

    pred_df = pd.DataFrame.from_records(predictions, columns=columns)
    with open('submission.csv', 'w') as file:
        pred_df.to_csv(file, index=False)


print('Data processing')
start = time.time()

train_df = pd.read_csv('data/train.csv')
tweets_df = preprocess(train_df)
tweets_df = tweets_df.reindex(np.random.permutation(tweets_df.index))
Y = tweets_df.loc[:, 'label']
X = tweets_df.loc[:, 'tweet']

end = time.time()
print(end-start)


num_ex = len(Y)
split = int(split_per * num_ex)

print('TF-IDF embeddings')
start = time.time()

vectorizer = TfidfVectorizer(max_features=8000)
tf_idf_matrix = vectorizer.fit_transform(X)

end = time.time()
print(end-start)


print('Logistic Regression model training')
start = time.time()

X_train = tf_idf_matrix[:split]
X_test = tf_idf_matrix[split:]
Y_train, Y_test = Y[:split], Y[split:]
logreg = linear_model.LogisticRegression(C=1)     # C = Regularization coefficient
logreg.fit(X_train,Y_train)

end = time.time()
print(end-start)


print('Prediction')

pred_train = logreg.predict(X_train)
pred_test = logreg.predict(X_test)

# Weighted F1 score
accuracy_train = f1_score(Y_train, pred_train, average='weighted')
accuracy_test = f1_score(Y_test, pred_test, average='weighted')
print("Training Weighted F1 score: %s" % accuracy_train)
print("Test Weighted F1 score: %s" % accuracy_test)


print('Test Data processing')
start = time.time()

test_df = pd.read_csv('data/test.csv')
tweets_test_df = test_preprocess(test_df)
ID_test = tweets_test_df.loc[:, 'id']
X_new = tweets_test_df.loc[:, 'tweet']

end = time.time()
print(end-start)


print('Test Data prediction')
test_data_prediction(X_new, ID_test)
