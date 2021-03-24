import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.tokens import Doc
import pandas as pd
import numpy as np
from random import sample
import io, csv
import re
import random
import json
from tqdm import tqdm
nlp = spacy.load('de_core_news_lg')
#nlp = spacy.load('de_dep_news_trf')
from pprint import pprint


def load_training_data(infile):
    with open(infile) as json_file:
        sentences = json.load(json_file)
    print("LENGTH OF DATASET: ", len(sentences))
    
    dataset_dict, entity_counter = dict(), dict()
    for sentence in sentences:
        entities = sentence[1]["entities"]
        for entity in entities:
            label = entity[2]
            try:
                entity_counter[label] += 1
            except KeyError:
                entity_counter[label] = 1
    pprint(entity_counter)
    return sentences

def load_retrieval_data(infile):
    with open(infile, 'r') as inf:
        data = pd.read_csv(inf)
    return data
    





if __name__ == "__main__":
    dataset = load_training_data('input/tagged_sentences_latest.json')
    pprint(dataset[:10])
    #{'GRAN': 1975, 'LEVEL_LOC': 265, 'SINGLE_LOC': 1035, 'TIME': 1363}
    
    retrieval_data = load_retrieval_data('input/info_retrieval_data_latest.csv')
    