import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegression
import csv
import sys
import random
from sklearn.metrics import classification_report

class Sampler(object):
    def __init__(self):
        self.unlabeled = []
        self.labeled = []
        self.labels = []
        with open(r"UnlabeledEDUS1.txt") as infile:
            for line in infile:  
                self.unlabeled.append(line)
        with open(r"labeledEDUS.txt") as infile: 
            for line in infile: 
                array = line.split(" ")
                if("<negative>" in array): 
                    self.labels.append(0)
                elif("<positive>" in array): 
                    self.labels.append(0)
                elif("<neutral>" in array): 
                    self.labels.append(1)
                else: 
                    continue
                self.labeled.append(line)
    def sample(self, k): 
        pass 
    def save(self): 
        with open(r"UnlabeledEDUS1.txt", "w") as f: 
            for line in self.unlabeled: 
                f.write(line)  
        with open(r"LabeledEDUS.txt", "w") as f: 
            for line in self.labeled: 
                f.write(line) 
                
    def process_k_edus(self, k_indices): 
        for k in k_indices: 
            print(self.unlabeled[k]) 
            label = input("Please label this EDU: ")
            if label=="nega":
                label="negative"
            elif label=="p":
                label="positive"
            elif label=="neut":
                label="neutral"
            if(label=="negative"): 
                self.labels.append(0)
            elif("positive"==label): 
                self.labels.append(0)
            elif("neutral"== label): 
                self.labels.append(1)
            else: 
                print("That label is not accepted") 
                continue
            i = self.unlabeled[k].index('\n') 
            edu = self.unlabeled[k][:i] + ' <' + label + '> \n'
            self.labeled.append(edu)
            
        for i in sorted(k_indices, reverse=True): 
            del self.unlabeled[i]
    def give_unlabeled_list(self):
        return self.unlabeled
    def give_labeled_list(self):
        return self.labeled
    
 
class RandomSampler(Sampler): 
    def sample(self,k): 
        k_indices = random.sample(range(0, len(self.unlabeled)),k)
        return k_indices
 
class UncertaintySampler(Sampler): 
    def sample(self,k): 
        X_labeled, y_labels, X_unlabeled= self.create()
        logreg = LogisticRegression(C=1, solver = 'lbfgs', warm_start=True)
        logreg.fit(X_labeled,y_labels) 
        probabilities = logreg.predict_proba(X_unlabeled)
        p_neutral = probabilities[:,1]
        p = []
        for x in p_neutral: 
            p.append(abs(x-0.5)) 
        prob_neutral = np.argsort(p) 
        k_indices = [] 
        for i in range(k): 
            k_indices.append(prob_neutral[i])
        return k_indices 
    def create(self): 
        X_train_corpus=[]
        y_labels = []
        for line in self.labeled:
            array = line.split(" ")
            if("<negative>" in array): 
                y_labels.append(-1)
            elif("<positive>" in array): 
                y_labels.append(1)
            elif("<neutral>" in array): 
                continue               
            i = line.find('<')
            line = line[:i]
            X_train_corpus.append(line)
    
        X_test_corpus=self.unlabeled 
        token = r"(?u)\b[\w\'/]+\b"
        tf_vectorizer = CountVectorizer(lowercase=True, max_df=1.0, min_df=1, binary=True, token_pattern=token)
        tf_vectorizer.set_params(ngram_range=(1,1))
        X_labeled = tf_vectorizer.fit_transform(X_train_corpus)
        X_unlabeled = tf_vectorizer.transform(X_test_corpus)
        return X_labeled, y_labels, X_unlabeled, X_train_corpus, X_test_corpus
 
##rs = UncertaintySampler() 
##cont = "T" 
##while (cont=="T"): 
##    k_indices = rs.sample(21)
##    rs.process_k_edus(k_indices)
##    rs.save()  
##    cont = input("continue? T/F ")
## 

## labeled_data=[]
## data_labels= []
## neut = 0 
## neg =0 
## pos = 0 
## for line in rs.labeled:
##     array = line.split(" ")
#     if("<negative>" in array): 
##        data_labels.append(-1)
##        neg = neg+1
##    elif("<positive>" in array): 
##        data_labels.append(1)
##        pos = pos+1
##    elif("<neutral>" in array): 
##        data_labels.append(0) 
##        neut = neut+1
##    i = line.find('<')
##    line = line[:i]
##    labeled_data.append(line)
## 
### human_terms = [] 
# with open(r"/Users/dorsazeinali/Desktop/imdb-unigrams.txt", 'r') as f:
#     for line in f: 
#         i = line.find('<')
#         line = line[:i]
#         human_terms.append(line)
 
## Labeling the human terms -1,1
# human_terms_label = [] 
# for term in human_terms: 
#     print(term)
#     label = input("Please label this term P/N")
#     human_terms_label.append(label)"
 
# human_labels = [] 
# for label in human_terms_label: 
#     if(label =="N"): 
#         human_labels.append(-1)
#     if(label=="P"): 
#         human_labels.append(1)
