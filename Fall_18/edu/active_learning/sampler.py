import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import random


class Sampler(object):
    """
    Sampler class

    Attributes
    ----------
    unlabeled : list of str
        list of unlabeled EDUs
    labeled: list of str
        list of labeled EDUs
    labels: list of int
        list of labels
    
    Methods
    -------
    sample(k)
        Not implemented in this base Sampler class
    save()
        write the attributes to the file
    process_k_edus(k_indices)
        handle the labelling process for the k indices passed
        
    """
    
    # dictionary to map str label to int for self.labels
    _is_neutral = { 
                    'n': 0, #negative 
                    'z': 1, #neutral
                    'p': 0  #positive 
                  } 
    
    def __init__(self):
        self.unlabeled = self.__read_unlabeled(r"UnlabeledEDUs.txt")
#         self.labeled, self.labels = self.__read_labeled(r"labeledEDUs.txt")
        self.labeled, self.labels = self.__read_labeled(r"test_random.txt")
        
    def sample(self, k): 
        raise AttributeError('base Sample class does not implement method sample()') 
        
    def save(self):
        """
        writes the attributes to their respective files
        """
        with open(r"UnlabeledEDUS.txt", "w", newline='') as f: 
            for line in self.unlabeled: 
                f.write(line)  
#         with open(r"LabeledEDUS.txt", "w", newline='') as f: 
#             for line in self.labeled: 
#                 f.write(line)
        with open(r"test_random.txt", "w", newline='') as f: 
            for line in self.labeled: 
                f.write(line) 
                
    def process_k_edus(self, k_indices): 
        """
        Label k EDU 

        Parameters
        ----------
        arg1 : list of int
            list of indices to the unlabeled EDUs to label
        """ 
        for k in k_indices:    
            # print EDU
            edu = self.unlabeled[k]
            print('EDU:')
            print('\t', edu)
        
            # get label input
            label = self.__sanitized_input("Please label this EDU: ( n = negative | z = neutral | p = positive )")
            
            # update labels and labeled data 
            self.labels.append(self._is_neutral[label])
            #self.labeled.append(edu + ' ' + label)
            self.labeled.append(edu[:-1] + ' ' + label + '\n' )
            
            print('======> LABELED', label.upper(), '\n')
            
        for i in sorted(k_indices, reverse=True): 
            del self.unlabeled[i] 
            
            
    def __sanitized_input(self, prompt):
        """
        Ask for input labels until valid input 

        Parameters
        ----------
        arg1 : str
            the input prompt
        """
        while True:
            uinput = input(prompt)
            uinput = uinput.lower()
            
            if uinput in ['n', 'neg', 'negative', '-1', '-']:
                return 'n'
            elif uinput in ['z', 'neu', 'neutral', '0', '_']:
                return 'z'
            elif uinput in ['p', 'pos', 'positive', '1', '+']:
                return 'p'
            else:
                print('===> ERR: provide valid label [n|z|p]')
                       
    # read functions  
    def __read_unlabeled(self, filename):
        """
        read the unlabeled EDUs file to fill in the unlabeled list

        Parameters
        ----------
        arg1 : str
            file name of unlabeled data

        Returns
        -------
        unlabeled: list of str
            list of unlabeled EDUs

        """
        unlabeled = []
        with open(filename) as infile: 
            for line in infile:  
                unlabeled.append(line)
    
        return unlabeled
    
    def __read_labeled(self, filename):
        """
        read the unlabeled EDUs file to fill in the unlabeled list

        Parameters
        ----------
        arg1 : str
            file name of unlabeled data

        Returns
        -------
        labeled: list of str
            list of labeled EDUs 
        labels: list of int
            list of labels
            
        """
        labeled = []
        labels = []
        
        with open(filename) as infile: 
            for line in infile: 
                #print(line)
                labels.append(self._is_neutral[line[-2]]) 
                labeled.append(line)
        
        return labeled, labels
        
        
        
"""
Class: Random Sampler
"""     
class RandomSampler(Sampler): 
    """
    Random Sampler class inherits Sampler
    
    Methods
    -------
    sample(k)
        get k random indices
        
    """
    def sample(self,k): 
        """
        sample k indices

        Parameters
        ----------
        arg1 : int
            number of samples required

        Returns
        -------
        k_indices: list of int
            list of k random indices

        """
        k_indices = random.sample(range(0, len(self.unlabeled)),k)
        return k_indices 
    
    
    
"""
Class: Uncertainty Sampler
"""         
class UncertaintySampler(Sampler):
    """
    Random Sampler class inherits Sampler
    
    Methods
    -------
    sample(k)
        get k indices that have uncertain probabilities (~ 0.5)
        
    create()
        generate vectorized data for sample()
        
    """
    
    # dictionary to map str label to int for self.labels
    _labels = { 
                'n': -1, #negative 
                'z':  0, #neutral
                'p':  1  #positive 
              }
    
    def sample(self,k): 
        """
        sample k indices by uncertainty calculated using logistic regression

        Parameters
        ----------
        arg1 : int
            number of samples required

        Returns
        -------
        k_indices: list of int
            list of k uncertain indices

        """
        # generate data
        X_labeled, y_labels, X_unlabeled= self.create()
        
        # Logistic Regression 
        logreg = LogisticRegression(C=1, solver = 'lbfgs', warm_start=True)
        logreg.fit(X_labeled,y_labels) 
        probabilities = logreg.predict_proba(X_unlabeled)
        p_neutral = probabilities[:,1]
        p = []
        for x in p_neutral: 
            p.append(abs(x-0.5)) 
        
        # get the uncertain indices
        prob_neutral = np.argsort(p) 
        k_indices = [] 
        for i in range(k): 
            k_indices.append(prob_neutral[i])
            
        return k_indices
    
    def create(self): 
        """
        vectorizes the data
        
        Returns
        -------
        X_labeled: matrix of int
            vectorized labeled EDUs
        y_labels: list of int
            labels of labeled EDUs
        X_unlabeled: matrix of int
            vectorized unlabeled EDUs
        """
        X_train_corpus=[]
        y_labels = []
        for line in self.labeled:
            # the last character is the label
            y_labels.append(self._labels[line[-2]])
            # get rid of the last character
            line = line[:-2]
            X_train_corpus.append(line)
    
        X_test_corpus=self.unlabeled
        
        # vectorize the corpus
        token = r"(?u)\b[\w\'/]+\b"
        tf_vectorizer = CountVectorizer(lowercase=True, max_df=1.0, min_df=1, binary=True, token_pattern=token)
        tf_vectorizer.set_params(ngram_range=(1,1))
        
        X_labeled = tf_vectorizer.fit_transform(X_train_corpus)
        X_unlabeled = tf_vectorizer.transform(X_test_corpus)
        
        return X_labeled, y_labels, X_unlabeled     
    