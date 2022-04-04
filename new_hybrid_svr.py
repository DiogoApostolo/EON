

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np

# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict

from scipy.stats import pearsonr

import pickle
import csv
from gensim.test.utils import get_tmpfile

from sklearn.feature_extraction.text import TfidfVectorizer


with open("cv_lbls2.txt", "rb") as input_file:
       labels = pickle.load(input_file)

with open("test_lbls2.txt", "rb") as input_file:
       test_labels = pickle.load(input_file)

print(len(labels))

with open("swn_negative_cv_documents.txt", "rb") as input_file:
       swn_negative_cv_documents = pickle.load(input_file)

with open("swn_positive_cv_documents.txt", "rb") as input_file:
       swn_positive_cv_documents = pickle.load(input_file)

with open("pol_val_cv_documents.txt", "rb") as input_file:
       pol_val_cv_documents = pickle.load(input_file)


with open("swn_negative_test_documents.txt", "rb") as input_file:
       swn_negative_test_documents = pickle.load(input_file)

with open("swn_positive_test_documents.txt", "rb") as input_file:
       swn_positive_test_documents = pickle.load(input_file)

with open("pol_val_test_documents.txt", "rb") as input_file:
       pol_val_test_documents = pickle.load(input_file)

with open("cv_pre_processed_documments2.txt", "rb") as input_file:
       pre_processed_documments = pickle.load(input_file)


with open("test_pre_processed_documments2.txt", "rb") as input_file:
       test_pre_processed_documments = pickle.load(input_file)

print(len(swn_negative_cv_documents))
print(len(swn_positive_cv_documents))
print(len(pol_val_cv_documents))


vectors = []
max_len = 0
for i in range(len(swn_negative_cv_documents)):
      
   vector = []
   vector+=(swn_negative_cv_documents[i])
   vector+=(swn_positive_cv_documents[i])
   vector+=(pol_val_cv_documents[i])
   vectors.append(vector)
   if(max_len<len(vector)):
     max_len = len(vector)
     

test_vectors = []

for i in range(len(swn_negative_test_documents)):
      
   vector = []
   vector+=(swn_negative_test_documents[i])
   vector+=(swn_positive_test_documents[i])
   vector+=(pol_val_test_documents[i])
   test_vectors.append(vector)
   if(max_len<len(vector)):
     max_len = len(vector)
     print(max_len)


hybrid_vectors = []
for i in range(len(swn_positive_cv_documents)):
    hybrid_vector = vectors[i]
    hybrid_vector = np.pad(hybrid_vector, (0, max_len-len(vectors[i])), 'constant')
    hybrid_vectors.append(hybrid_vector)

hybrid_vectors_test = []
for i in range(len(swn_positive_test_documents)):
    hybrid_vector = vectors[i]
    hybrid_vector = np.pad(hybrid_vector, (0, max_len-len(vectors[i])), 'constant')
    hybrid_vectors_test.append(hybrid_vector)

new_docs = []
for doc in pre_processed_documments:
       new_string = ""
       for token in doc:
              new_string+=(token)
              new_string+=" "
       #print(new_string)
       new_docs.append(new_string)

new_test_docs = []
for doc in test_pre_processed_documments:
       new_string = ""
       for token in doc:
              new_string+=(token)
              new_string+=" "
       #print(new_string)
       new_test_docs.append(new_string)

vectorizer = TfidfVectorizer(max_features=2000)
vectors = vectorizer.fit_transform(new_docs)
test_vectors = vectorizer.transform(new_test_docs)

vectors = vectors.toarray().astype(float)
test_vectors = test_vectors.toarray().astype(float)

print(len(hybrid_vectors))

new_vectors = []
for i in range(len(vectors)):
    new_vector = np.concatenate((vectors[i],hybrid_vectors[i]))
    new_vectors.append(new_vector)


new_vectors_test = []
for i in range(len(test_vectors)):
    new_vector = np.concatenate((test_vectors[i],hybrid_vectors_test[i]))
    new_vectors_test.append(new_vector)

    

print(len(new_vectors))
print(len(new_vectors_test))
rows = [["c_val","fold1","fold2","fold3","fold4","fold5","mean","std"]]


'''
cs = [0.0625,0.125,0.25,0.5,1,2,4,8,16]
epsilons = [0, 0.01, 0.1, 0.5, 1, 2, 4]
for i in cs:
       for e in epsilons:
              print(i)
              print(e)
              regr = make_pipeline(StandardScaler(), SVR(C=i, epsilon=e))
              cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
              #scores = cross_val_score(regr, vectors, labels, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
              
              ypred = cross_val_predict(regr, new_vectors, labels, cv=cv, n_jobs=-1)

              # report performance
              
              #print(len(ypred))
              
              print(mean_squared_error(labels,ypred))
              print(mean_absolute_error(labels,ypred))
              print(pearsonr(labels, ypred))
              
              #print(scores)
              #print('MSE: %.3f (%.3f)' % (mean(scores), std(scores)))

              rows.append([i,e,mean_squared_error(labels,ypred),mean_absolute_error(labels,ypred),pearsonr(labels, ypred)])
              



results = open("results_new_hybrid.csv","w",newline='')


writer = csv.writer(results)

#new_rows = header + rows
writer.writerows(rows)
'''

regr = make_pipeline(StandardScaler(), SVR(C=4.0, epsilon=0.01))
regr.fit(new_vectors, labels)

ypred=regr.predict(new_vectors_test)


print(mean_squared_error(test_labels,ypred))
print(mean_absolute_error(test_labels,ypred))
print(pearsonr(test_labels, ypred))