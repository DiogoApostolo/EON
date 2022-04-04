from sklearn.metrics.regression import mean_absolute_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict

import pickle

from scipy.stats import pearsonr

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
import csv
with open("cv_lbls2.txt", "rb") as input_file:
        labels = pickle.load(input_file)

with open("test_lbls2.txt", "rb") as input_file:
       test_labels = pickle.load(input_file)

'''
with open("vectors.txt", "rb") as input_file:
        vectors = pickle.load(input_file)
'''


'''
with open("sentiment_score_reviews.txt", "rb") as input_file:
        sentiment_score_reviews = pickle.load(input_file)
'''

with open("sentiment_score_cv_documents.txt", "rb") as input_file:
        sentiment_score_sentiment_reviews = pickle.load(input_file)

with open("sentiment_score_test_documents.txt", "rb") as input_file:
        sentiment_score_test_reviews = pickle.load(input_file)

with open("cv_pre_processed_documments2.txt", "rb") as input_file:
       pre_processed_documments = pickle.load(input_file)

with open("test_pre_processed_documments2.txt", "rb") as input_file:
       test_pre_processed_documments = pickle.load(input_file)

#vectors = vectors[0:1000]
#labels = labels[0:1000]






size = {}


docs = []
sentiment_docs = []
lbls = []
for i in range(len(sentiment_score_sentiment_reviews)):

        text_doc = pre_processed_documments[i]
        doc=sentiment_score_sentiment_reviews[i]

        if(len(doc)<=60):
                docs.append(text_doc)
                sentiment_docs.append(doc)
                lbls.append(labels[i])


        if str(len(doc)) in size:
                size[str(len(doc))]+=1
        else:   
                size[str(len(doc))]=1

pre_processed_documments = docs
sentiment_score_sentiment_reviews = sentiment_docs
labels = lbls



docs = []
sentiment_docs = []
lbls = []
for i in range(len(sentiment_score_test_reviews)):

        text_doc = test_pre_processed_documments[i]
        doc=sentiment_score_test_reviews[i]

        if(len(doc)<=60):
                docs.append(text_doc)
                sentiment_docs.append(doc)
                lbls.append(test_labels[i])


        if str(len(doc)) in size:
                size[str(len(doc))]+=1
        else:   
                size[str(len(doc))]=1

test_pre_processed_documments = docs
sentiment_score_test_reviews = sentiment_docs
test_labels = lbls


new_docs = []
for doc in pre_processed_documments:
       new_string = ""
       for token in doc:
              new_string+=(token)
              new_string+=" "
       #print(new_string)
       new_docs.append(new_string)


test_new_docs = []
for doc in test_pre_processed_documments:
       new_string = ""
       for token in doc:
              new_string+=(token)
              new_string+=" "
       #print(new_string)
       test_new_docs.append(new_string)


vectorizer = TfidfVectorizer(max_features=2000)
vectors = vectorizer.fit_transform(new_docs)
vectors = vectors.toarray().astype(float)

test_vectors = vectorizer.transform(test_new_docs)
test_vectors = test_vectors.toarray().astype(float)


print(size)
MAX_SIZE_SENTENCE = 60
hybrid_vectors = []
i = 0
for vector in vectors:
    
    for s in sentiment_score_sentiment_reviews[i]:
        hybrid_vector = np.append(vector, s)
        vector = hybrid_vector
    
    
    #hybrid_vector = sentiment_score_sentiment_reviews[i]
    hybrid_vector = np.pad(hybrid_vector, (0, MAX_SIZE_SENTENCE-len(sentiment_score_sentiment_reviews[i])), 'constant')
   
    hybrid_vectors.append(hybrid_vector)
    i+=1



test_hybrid_vectors = []
i = 0
for vector in test_vectors:
    
    for s in sentiment_score_test_reviews[i]:
        hybrid_vector = np.append(vector, s)
        vector = hybrid_vector
    
    
    #hybrid_vector = sentiment_score_test_reviews[i]
    hybrid_vector = np.pad(hybrid_vector, (0, MAX_SIZE_SENTENCE-len(sentiment_score_test_reviews[i])), 'constant')
   
    test_hybrid_vectors.append(hybrid_vector)
    i+=1

'''
rows = [["c_val","fold1","fold2","fold3","fold4","fold5","mean","std"]]
cs = [0.0625,0.125,0.25,0.5,1,2,4,8,16]



epsilons = [0, 0.01, 0.1, 0.5, 1, 2, 4]
for i in cs:
       for e in epsilons:
              print(i)
              print(e)
              regr = make_pipeline(StandardScaler(), SVR(C=i, epsilon=e))
              cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
              #scores = cross_val_score(regr, vectors, labels, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
              
              ypred = cross_val_predict(regr, hybrid_vectors, labels, cv=cv, n_jobs=-1)

              # report performance
              
              #print(len(ypred))
              
              print(mean_squared_error(labels,ypred))
              print(mean_absolute_error(labels,ypred))
              print(pearsonr(labels, ypred))
              
              #print(scores)
              #print('MSE: %.3f (%.3f)' % (mean(scores), std(scores)))

              rows.append([i,e,mean_squared_error(labels,ypred),mean_absolute_error(labels,ypred),pearsonr(labels, ypred)])
              



results = open("results_hybrid_sentences.csv","w",newline='')


writer = csv.writer(results)

#new_rows = header + rows
writer.writerows(rows)
'''
print(len(hybrid_vectors))
print(len(test_hybrid_vectors))

regr = make_pipeline(StandardScaler(), SVR(C=4.0, epsilon=0.01))
regr.fit(hybrid_vectors, labels)

ypred=regr.predict(test_hybrid_vectors)


print(mean_squared_error(test_labels,ypred))
print(mean_absolute_error(test_labels,ypred))
print(pearsonr(test_labels, ypred))