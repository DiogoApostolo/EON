from joblib.logger import PrintTime
import nltk
import csv
import re
from nltk.util import Index
from numpy import multiply
import rdflib
from rdflib import Graph, URIRef, Literal
from gensim.models import word2vec
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle


def sentiment_polarity_tokens_mean(sentiment_score_sentence):
    
    if len(sentiment_score_sentence)==0:
        sentiment_score_final=0.5
    else:
        sentiment_score_final=sum(sentiment_score_sentence)/len(sentiment_score_sentence)
    
    return sentiment_score_final

def sentiment_polarity_tokens_max(sentiment_score_sentence):
    
    sentiment_score_final = 0
    if len(sentiment_score_sentence)==0:
            sentiment_score_final=0.5
    else:
        for sentiment_score in sentiment_score_sentence:
            abs_sentiment_score = abs(sentiment_score-0.5)
            if abs_sentiment_score>sentiment_score_final:
                sentiment_score_final=sentiment_score
    
    return sentiment_score_final


def sentiment_polarity_tokens_several_max(sentiment_score_sentence, num_maxes):
    
    sentiment_score_final = []
    if len(sentiment_score_sentence)==0:
            sentiment_score_final=0.5
    else:
        sentiment_score_sentence.sort(reverse = True ,key = my_metric)
        if len(sentiment_score_sentence)>num_maxes:
            sentiment_score_final_lst = sentiment_score_sentence[0:num_maxes]
        else:
            sentiment_score_final_lst = sentiment_score_sentence[0:len(sentiment_score_sentence)]


        sentiment_score_final = sum(sentiment_score_final_lst)/len(sentiment_score_final_lst)
    
    return sentiment_score_final

def my_metric(e):
    return abs(e-0.5)

def sentiment_polarity_tokens_neg_weight(sentiment_score_sentence):
    
    if len(sentiment_score_sentence)==0:
        sentiment_score_final=0.5
    else:
        sentiment_score_sentence_2 =[]
        for sentiment_score in sentiment_score_sentence:
            sentiment_score_sentence_2.append(sentiment_score)
            if sentiment_score < 0.5:
                sentiment_score_sentence_2.append(sentiment_score)
        sentiment_score_final=sum(sentiment_score_sentence_2)/len(sentiment_score_sentence_2)
    
    return sentiment_score_final

def sentiment_polarity_tokens_neg_weight_3(sentiment_score_sentence):
    
    if len(sentiment_score_sentence)==0:
        sentiment_score_final=0.5
    else:
        sentiment_score_sentence_2 =[]
        for sentiment_score in sentiment_score_sentence:
            sentiment_score_sentence_2.append(sentiment_score)
            if sentiment_score < 0.5:
                sentiment_score_sentence_2.append(sentiment_score)
                sentiment_score_sentence_2.append(sentiment_score)
        sentiment_score_final=sum(sentiment_score_sentence_2)/len(sentiment_score_sentence_2)
    
    return sentiment_score_final


def sentiment_polarity_tokens_neg_weight_4(sentiment_score_sentence):
    
    if len(sentiment_score_sentence)==0:
        sentiment_score_final=0.5
    else:
        sentiment_score_sentence_2 =[]
        for sentiment_score in sentiment_score_sentence:
            sentiment_score_sentence_2.append(sentiment_score)
            if sentiment_score < 0.5:
                sentiment_score_sentence_2.append(sentiment_score)
                sentiment_score_sentence_2.append(sentiment_score)
                sentiment_score_sentence_2.append(sentiment_score)
        sentiment_score_final=sum(sentiment_score_sentence_2)/len(sentiment_score_sentence_2)
    
    return sentiment_score_final

def sentiment_classifier_basic(sentence, g, polarity_dic,cycle_num):
    #Classifies sentences's sentiment polarity into positive(1) or negative(0)

    sentiment_score_sentence = [] 
    itr =0
    for token in sentence:
        if token in polarity_dic:
            sentiment_score = polarity_dic.get(token)

            
            if sentiment_score>=0.2:
                sentiment_score_sentence.append(sentiment_score)
        else:
            knows_query = '''
                SELECT ?SenticConcept ?text ?polarity ?SWN_Positive ?SWN_Negative
                WHERE {
                    ?SenticConcept :text ?text.
                    ?SenticConcept :text \"''' + token +'''\".
                    ?SenticConcept :polarity ?polarity .
                    ?SenticConcept :SWN_Positive ?SWN_Positive.
                    ?SenticConcept :SWN_Negative ?SWN_Negative.
                }
            '''

            qres = g.query(knows_query)
            
            for row in qres:
                
                pol_val= float(row.polarity)
                swn_positive = float(row.SWN_Positive)
                swn_negative = float(row.SWN_Negative)

                #Classifies sentiment of token
                sentiment_score = (pol_val+(swn_positive-swn_negative))/2

                #sentiment_score =  pol_val
                

                sentiment_score = (sentiment_score+1)/2

                if abs(sentiment_score-0.5)>=0.2:
                    sentiment_score_sentence.append(sentiment_score)
                
                polarity_dic[token] = sentiment_score


            #itr=itr+1;
    
    if cycle_num == 0:
        sentiment_score_final = sentiment_polarity_tokens_mean(sentiment_score_sentence)
    elif cycle_num == 1:
        sentiment_score_final = sentiment_polarity_tokens_max(sentiment_score_sentence)
    elif cycle_num == 2:
        sentiment_score_final = sentiment_polarity_tokens_several_max(sentiment_score_sentence,3)
    elif cycle_num == 3:
        sentiment_score_final = sentiment_polarity_tokens_several_max(sentiment_score_sentence,5)
    elif cycle_num == 4:
        sentiment_score_final = sentiment_polarity_tokens_neg_weight(sentiment_score_sentence)
    elif cycle_num == 5:
        sentiment_score_final = sentiment_polarity_tokens_neg_weight_3(sentiment_score_sentence)
    elif cycle_num == 6:
        sentiment_score_final = sentiment_polarity_tokens_neg_weight_4(sentiment_score_sentence)

    #sentiment_score_final = sentiment_polarity_tokens_max(sentiment_score_sentence)
    #sentiment_score_final = sentiment_polarity_tokens_several_max(sentiment_score_sentence,3)

    return [sentiment_score_final]


def get_rating_distibituion(labels):
    rating_distibituion = []
    rating_distibituion.append(0)
    for i in range(10):
        inx = labels.count(i+1)
        if i==0:
            inx=inx-1
        rating_distibituion.append(rating_distibituion[-1]+inx/(len(labels)-1))
    
    return rating_distibituion

def get_review_score(score,rating_distribution):
    review_score=0
    for i in range(len(rating_distribution)):
        if rating_distribution[i]>=score:
            m = rating_distribution[i]-rating_distribution[i-1]
            b = rating_distribution[i]-m*i

            review_score = (score-b)/m
            break

    return review_score



    



def main():
    
    g = Graph()

    # Parse in an RDF file hosted on the Internet
    g.parse("output.owl", format='xml')

    with open("test_lbls2.txt", "rb") as input_file:
        labels = pickle.load(input_file)

    with open("test_pre_processed_documment_sentences2.txt", "rb") as input_file:
        pre_processed_documments = pickle.load(input_file)

    polarity_dic = {}

    print("read data")


    rating_distribution = get_rating_distibituion(labels)

    normal_distibution = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    #Drop pos reviews

    #neg_indx = [idx for idx, element in enumerate(labels) if element < 5]
    #labels = [labels[index] for index in neg_indx]
    #pre_processed_documments = [pre_processed_documments[index] for index in neg_indx]
    

    sentiment_score_reviews = []

    tp = 0
    fp = 0
    fn = 0
    tn = 0  
    itr = 0

    '''
    tokens_size_array= []
    for review in pre_processed_documments:
        tokens_size_array.append(sum(len(x) for x in review))

    print(max(tokens_size_array))
    '''

    i = 0
    for cycle_num in range(7):
        sentiment_score_reviews = []

        tp = 0
        fp = 0
        fn = 0
        tn = 0  
        itr = 0

        i = 0
        for review in pre_processed_documments:
            sentiment_score_sentences = []

            for sentence in review:
                #Get sentiment class of sentence
                [sentiment_score_sentence]  =  sentiment_classifier_basic(sentence,g,polarity_dic,cycle_num)
                
                if abs(sentiment_score_sentence-0.5)>=0.1:
                    sentiment_score_sentences.append(sentiment_score_sentence)
                
                



            sentiment_score_review = sentiment_polarity_tokens_neg_weight_4(sentiment_score_sentences)

            sentiment_score_review = get_review_score(sentiment_score_review,rating_distribution)

            sentiment_score_reviews.append(sentiment_score_review)

            review_text=[' '.join(sentence) for sentence in review]
            review_text='.'.join(review_text)

            i+=1
            if i%100==0:
                print(i)

            #print("Review: \n"+review_text)
            #print("Score: \n"+ str(sentiment_score_review))

        if cycle_num == 0:
            print("Neg 4 Mean Token\n")
        elif cycle_num == 1:
            print("Neg 4 Sentence Max Token\n")
        elif cycle_num == 2:
            print("Neg 4 Sentence Several Max 3 Token\n")
        elif cycle_num == 3:
            print("Neg 4 Sentence Several Max 5 Token\n")
        elif cycle_num == 4:
            print("Neg 4 Sentence Neg Token\n")
        elif cycle_num == 5:
            print("Neg 4 Sentence Neg 3 Token\n")
        elif cycle_num == 6:
            print("Neg 4 Sentence Neg 4 Token\n")
                
        mse_score = mean_squared_error(labels, sentiment_score_reviews)
        print("Mean Square: \n"+str(mse_score))

        mae_score = mean_absolute_error(labels, sentiment_score_reviews)
        print("Mean Absolute: \n"+str(mae_score))

        corr_score = pearsonr(labels, sentiment_score_reviews)
        print("Correlation: \n"+str(corr_score))

    '''    
    accuracy = (tn+tp)/(tn+fp+fn+tp) 
    precision = tp/(tp+fp)	
    recall = tp/(tp+fn)
    f1_score = (2*tp)/(2*tp+fp+fn)

    #Print metric
    print("\n\n\n\n\n\n")
    print("Total accuracy: "+ str(accuracy))
    print("Total precision: "+ str(precision))
    print("Total recall: "+ str(recall))
    print("Total f1_score: "+ str(f1_score))
    '''


    #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    #model = word2vec.Word2Vec(pre_processed_sentences_all)
    #words = list(model.wv.index_to_key)
    #print(words)

    #vec = {word:model.wv[word] for word in words}

    #print(vec)
    '''
    entities = nltk.chunk.ne_chunk(tagged)

    print(entities)
    '''


main()

