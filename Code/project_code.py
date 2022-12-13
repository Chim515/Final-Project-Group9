# -*- coding: utf-8 -*-

import numpy as np
import random
import re, string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection, naive_bayes, svm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('globalterrorismdb_0718dist.csv',encoding="ISO-8859-1")

df.head()

df.isnull().sum()

df = df[['summary','attacktype1_txt']]
df.head()

df.isnull().sum()

df.dropna(axis=0,inplace=True)

df.isnull().sum()

len(df)

df['attacktype1_txt'].value_counts()

# CLEANING TEXT 
stopwords = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
# %%
# PREPROCESSING 

def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    combined_text = ' '.join(tokenized_text)
    text = [wordnet_lemmatizer.lemmatize(word) for word in combined_text if word not in stopwords]
    return combined_text

df['summary'] = df['summary'].apply(str).apply(lambda x: clean_text(x))

df['summary'] = df['summary'].apply(str).apply(lambda x: text_preprocessing(x))

# splitting the data
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['summary'],df['attacktype1_txt'],test_size=0.3)

# transforming the feature
Tfidf_vect = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
Tfidf_vect.fit(df['summary'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#Encoding target 
from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# handling class imbalance
from imblearn.over_sampling import SMOTE
sm = SMOTE()
Train_X_sm, Train_Y_sm = sm.fit_resample(Train_X_Tfidf,Train_Y)

pd.Series(Train_Y_sm).value_counts()

# Training the Logistic Regression classifier
logreg = LogisticRegression()
logreg.fit(Train_X_Tfidf,Train_Y)
# Testing the classifier
y_pred = logreg.predict(Test_X_Tfidf)
#print('Predicted',y_pred)
#print('Actual data',Test_Y)
print('Accuracy: {:.2f}'.format(accuracy_score(Test_Y, y_pred) * 100))

#print('Predicted probability',y_pred_proba)

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
print('Precision                                   : %.3f'%precision_score(Test_Y, y_pred, average='macro'))
print('Recall                                      : %.3f'%recall_score(Test_Y, y_pred, average='macro'))
print('F1-Score                                    : %.3f'%f1_score(Test_Y, y_pred, average='macro'))

n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    logreg = LogisticRegression(max_iter=n_iter)
    logreg.fit(Train_X_Tfidf,Train_Y)
    scores.append(logreg.score(Test_X_Tfidf, Test_Y))
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores)

from sklearn.svm import SVC  
clf = SVC(kernel='linear')
clf.fit(Train_X_Tfidf,Train_Y)
y_pred = clf.predict(Test_X_Tfidf)

# Training SVM Classifier 
clf = SGDClassifier(loss="hinge")
clf.fit(Train_X_Tfidf,Train_Y)
y_pred = clf.predict(Test_X_Tfidf)
print('Accuracy: {:.2f}'.format(accuracy_score(Test_Y, y_pred) * 100))

print('Precision                                   : %.3f'%precision_score(Test_Y, y_pred, average='macro'))
print('Recall                                      : %.3f'%recall_score(Test_Y, y_pred, average='macro'))
print('F1-Score                                    : %.3f'%f1_score(Test_Y, y_pred, average='macro'))

n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    clf = SGDClassifier(loss="hinge", max_iter=n_iter)
    clf.fit(Train_X_Tfidf,Train_Y)
    scores.append(clf.score(Test_X_Tfidf, Test_Y))
  
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores)

# Logistic Regression using SGD learning
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(Train_X_Tfidf,Train_Y)
y_pred = clf.predict(Test_X_Tfidf)
print('Accuracy: {:.2f}'.format(accuracy_score(Test_Y, y_pred) * 100))

print('Precision                                   : %.3f'%precision_score(Test_Y, y_pred, average='macro'))
print('Recall                                      : %.3f'%recall_score(Test_Y, y_pred, average='macro'))
print('F1-Score                                    : %.3f'%f1_score(Test_Y, y_pred, average='macro'))

# SVM using SGD learning
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(Train_X_Tfidf,Train_Y)
y_pred = clf.predict(Test_X_Tfidf)
print('Accuracy: {:.2f}'.format(accuracy_score(Test_Y, y_pred) * 100))

print('Precision                                   : %.3f'%precision_score(Test_Y, y_pred, average='macro'))
print('Recall                                      : %.3f'%recall_score(Test_Y, y_pred, average='macro'))
print('F1-Score                                    : %.3f'%f1_score(Test_Y, y_pred, average='macro'))

losses = ["hinge", "log", "perceptron"]
scores = []
for loss in losses:
    clf = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    clf.fit(Train_X_Tfidf,Train_Y)
    scores.append(clf.score(Test_X_Tfidf, Test_Y))
  
plt.title("Effect of loss")
plt.xlabel("loss")
plt.ylabel("score")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores)

from sklearn.preprocessing import MaxAbsScaler

# data values

# transofrm data
scaler = MaxAbsScaler()
rescaledX = scaler.fit_transform(Train_X_Tfidf)
rescaledX_test = scaler.fit_transform(Test_X_Tfidf)

from sklearn.model_selection import GridSearchCV

params = {
    "loss" : ["hinge", "log"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1"],
}

clf = SGDClassifier(max_iter=1000)
grid = GridSearchCV(clf, param_grid=params, cv=10)


grid.fit(rescaledX, Train_Y)

print(grid.best_params_)

grid_predictions = grid.predict(rescaledX_test) 

print('Accuracy: {:.2f}'.format(accuracy_score(Test_Y, grid_predictions)))

print('Precision                                   : %.3f'%precision_score(Test_Y, grid_predictions, average='macro'))
print('Recall                                      : %.3f'%recall_score(Test_Y, grid_predictions, average='macro'))
print('F1-Score                                    : %.3f'%f1_score(Test_Y, grid_predictions, average='macro'))
