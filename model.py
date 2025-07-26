# model.py
import nltk
from nltk.corpus import names
import random

def gender_features(word):
    return {'last_letter': word[-1]}

def train_model():
    data = [(name, 'male') for name in names.words('male.txt')] + \
           [(name, 'female') for name in names.words('female.txt')]
    random.shuffle(data)
    featuresets = [(gender_features(n), g) for (n, g) in data]
    train_data = featuresets[500:]
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    return classifier

# Train on import
classifier = train_model()
