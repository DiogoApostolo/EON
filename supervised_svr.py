

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

print(len(labels))

with open("cv_pre_processed_documments2.txt", "rb") as input_file:
       pre_processed_documments = pickle.load(input_file)

with open("test_lbls2.txt", "rb") as input_file:
       test_labels = pickle.load(input_file)

print(len(labels))

with open("test_pre_processed_documments2.txt", "rb") as input_file:
       test_pre_processed_documments = pickle.load(input_file)


print(len(pre_processed_documments))

print("read data")


#simplified version
#pre_processed_documments = pre_processed_documments[0:1000]


#labels  = labels[0:1000]

new_labels = labels


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(pre_processed_documments)]



#model = Doc2Vec.load("emb_model")

#-----------------------Vectorization-----------------------#
model = Doc2Vec(documents, vector_size=200, window=2, min_count=1, workers=4, epochs=200)
model.save("emb_model2_200")

words = list(model.wv.index_to_key)

vectors = []
for doc in pre_processed_documments:
    vector = model.infer_vector(doc)
    
    vectors.append(vector)

test_vectors = []
for doc in test_pre_processed_documments:
    vector = model.infer_vector(doc)
    
    test_vectors.append(vector)




with open("vectors2_200_500.txt","wb") as fp:
    pickle.dump(vectors,fp)


with open("test_vectors2_200_500.txt","wb") as fp:
    pickle.dump(test_vectors,fp)
#---------------------------End Vectorization---------------------------------#



#uncomment this and comment the previous section if you have a saved vectorization!
'''

with open("vectors2_200_500.txt","rb") as input_file:
       vectors = pickle.load(input_file)

with open("test_vectors2_200_500.txt","rb") as input_file:
       test_vectors = pickle.load(input_file)

'''

'''
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

#print(len(vectors))



print("embeding done")


#join with

'''


#-----------------------------CROSS VALIDATION-----------------------------------

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
              
              ypred = cross_val_predict(regr, vectors, new_labels, cv=cv, n_jobs=-1)

              # report performance
              
              #print(len(ypred))
              
              print(mean_squared_error(new_labels,ypred))
              print(mean_absolute_error(new_labels,ypred))
              print(pearsonr(new_labels, ypred))
              
              #print(scores)
              #print('MSE: %.3f (%.3f)' % (mean(scores), std(scores)))

              rows.append([i,e,mean_squared_error(new_labels,ypred),mean_absolute_error(new_labels,ypred),pearsonr(new_labels, ypred)])
              



results = open("results_3c2.csv","w",newline='')


writer = csv.writer(results)

#new_rows = header + rows
writer.writerows(rows)



#------------------------------ END CROSS VALIDATION ---------------------




#uncomment this if you want to test the SVR on the held of partition with specific SVR parameters
'''

#change this paremeters to the ones best obtained in the cross validation step
regr = make_pipeline(StandardScaler(), SVR(C=4.0, epsilon=0.1))
regr.fit(vectors, labels)

ypred=regr.predict(test_vectors)


print(mean_squared_error(test_labels,ypred))
print(mean_absolute_error(test_labels,ypred))
print(pearsonr(test_labels, ypred))
'''