import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def load_labeled(path, shuffle=True, random_state=42):
    import glob 
    labeled_file = glob.glob(path+"/LabeledEDUS.txt") 
    
    X_corpus = []
    y = []
    
    f = open(labeled_file[0], 'r', encoding="utf8")
    doc = f.read()
    for line in doc.split("\n"):
        if len(line) < 2:
            continue
        if line[-1] == "z":
            pass
        else:
            X_corpus.append(line[:-1])
            if line[-1] == "p":
                y.append(1)
            else:
                y.append(0)
    f.close()
    

    print("Labeled Data loaded.")
    
    y = np.array(y)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y))       
        
        X_corpus = [X_corpus[i] for i in indices]
        y = y[indices]
        
        
    return X_corpus, y

def load_labeled_neutrals(path, shuffle=True, random_state=42):
    import glob 
    labeled_file = glob.glob(path+"/LabeledEDUS.txt") 
    
    X_corpus = []
    y = []
    
    f = open(labeled_file[0], 'r', encoding="utf8")
    doc = f.read()
    for line in doc.split("\n"):
        if len(line) < 2:
            continue

        X_corpus.append(line[:-1])
        if line[-1] == "p":
            y.append(1)
        elif line[-1]== "z":
            y.append(0)
        else:
            y.append(-1)
    f.close()
    

    print("Labeled Data loaded.")
    
    y = np.array(y)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y))       
        
        X_corpus = [X_corpus[i] for i in indices]
        y = y[indices]
        
    # ADD TEST DATA

    test_file = glob.glob(path+"/test_random.txt") 
    
    test_corpus = []
    test_y = []
    
    f = open(test_file[0], 'r', encoding="utf8")
    doc = f.read()
    for line in doc.split("\n"):
        if len(line) < 2:
            continue

        test_corpus.append(line[:-1])
        if line[-1] == "p":
            test_y.append(1)
        elif line[-1]== "z":
            test_y.append(0)
        else:
            test_y.append(-1)
    f.close()
    

    print("Test Data loaded.")
    
    test_y = np.array(test_y)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(test_y))       
        
        test_corpus = [test_corpus[i] for i in indices]
        test_y = test_y[indices]
        
        
        
    return X_corpus, y, test_corpus, test_y

def load_unlabeled(path, shuffle=True, random_state=42):
    import glob 
    labeled_file = glob.glob(path+"/UnlabeledEDUs.txt") 
    
    X_corpus = []
    
    f = open(labeled_file[0], 'r', encoding="utf8")
    doc = f.read()
    for line in doc.split("\n"):
        if len(line) < 2:
            continue

        X_corpus.append(line[:-1])

    f.close()
    

    print("Unlabeled Data loaded.")
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X_corpus))       
        
        X_corpus = [X_corpus[i] for i in indices]
        

    return X_corpus

def load_with_path(path=r"/Users/ekremguzelyel/Desktop/Assignments/Research/MLLab-IIT/edu/models/"):
   X_corpus , y= load_labeled(path)
   return X_corpus , y


def split_and_vectorize():
   # Load from path
   path=r"/Users/ekremguzelyel/Desktop/Assignments/Research/MLLab-IIT/edu/models/"
   X_corpus , y = load_labeled(path)
   X_train_corpus , X_test_corpus, y_train, y_test = train_test_split(X_corpus, y, test_size=1./3, random_state=42)

   # Vectorize the data
   token = r"(?u)\b[\w\'/]+\b"
   vectorizer = CountVectorizer(token_pattern=token, min_df=5, stop_words=["the","a","of","and","br","to"])

   X_train_vector = vectorizer.fit_transform(X_train_corpus)
   X_test_vector = vectorizer.transform(X_test_corpus)
   print("Data Vectorized")
   
   return X_train_vector , y_train, X_test_vector , y_test

def split_ngram(m,n):
   # Load from path
   path=r"/Users/ekremguzelyel/Desktop/Assignments/Research/MLLab-IIT/edu/models/"
   X_corpus , y = load_labeled(path)
   X_train_corpus , X_test_corpus, y_train, y_test = train_test_split(X_corpus, y, test_size=1./3, random_state=42)
    
   # Vectorize the data with ngram
   token = r"(?u)\b[\w\'/]+\b"
   vectorizer = CountVectorizer(token_pattern=token, min_df=5, stop_words=["the","a","of","and","br","to"], ngram_range=(m,n))

   X_train_vector = vectorizer.fit_transform(X_train_corpus)
   X_test_vector = vectorizer.transform(X_test_corpus)
   print("Data Vectorized with ngram")
   
   return X_train_vector , y_train, X_test_vector , y_test

def convert_to_sequence(vector):
    ''' Takes a vector and converts it into a sequence.
    Args:
        vector: An output of CountVectorizer. i.e X_train_vector, X_test_vector
    '''
    document_sequence = []

    for k in vector.toarray():
        sequence = []
        [sequence.append(i) for i,j in enumerate(k) if j]
        document_sequence.append(sequence)

    return document_sequence
