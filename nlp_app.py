import pandas as pd
import numpy as np
# Import and instantiate the Countvectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Build a classifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# Import training data into dataframe
df = pd.read_csv("data/train.csv")
# Format training X (tweet text) set for nlp model 
X_train = df.text.values

def vectorizer_fit():
    # Convert to a "bag of words"
    # Learn a vocabulary dictionary of all tokens in the raw documents / omit any tokens with less than 4 occurrences
    vectorizer = CountVectorizer(min_df=4).fit(X_train)
    # To create the "bag-of_words" representation we call the transform method / transform documents to document-term matrix.
    return vectorizer

def transform(vectorizer):
    X_train_vect = vectorizer.transform(X_train)
    return X_train_vect

def transform_tweet(vectorizer, tweet):
    vectorized_tweet = vectorizer.transform(tweet)
    return vectorized_tweet

def train_nlp_model(X_train_vect):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    # Try several values of the hyper-parameter C to optimize result
    param_grid = {"C":[0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(model, param_grid, cv=5)
    # Format training y (supervised outcomes "1" or "0") set for nlp model
    y_train = df['target'].values
    grid.fit(X_train_vect, y_train)
    print("Best cross-validation score: {:.3f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    return grid


def make_prediction(grid, vectorized_tweet):
    classified_outcome = grid.predict(vectorized_tweet)
    prediction = str(classified_outcome[0]).replace("0","Not a Disaster").replace("1","Disaster")
    return prediction


    

