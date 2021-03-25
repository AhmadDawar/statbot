import sys
import os
import pandas as pd
import spacy
import time
import pickle
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from tqdm import tqdm
from blingfire import text_to_words
import re
from pprint import pprint

nlp = spacy.load('de_core_news_lg', disable=["ner"])
nlp_tok = spacy.load('de_core_news_lg', disable=["tagger", "parser","ner"])

#local imports
from ner_training import load_training_data, load_retrieval_data

ignore_ents = ['LEVEL_LOC', 'SINGLE_LOC', 'TIME']


def _find_entities(tagged_data, retrieval_data):
    """
    For each tagged sentence, check if token is not tagged and a noun
    Save token and context for bm25 retrieval
    """
    ret = list()
    assert len(tagged_data) == retrieval_data.shape[0], "Please make sure that tagged data and retrieval data entries align for ID tagging."
    ids = retrieval_data.original_id.tolist()
    for i, (sentence, entity_dict) in enumerate(tagged_data):
        ignore_indices, relevant_tokens, relevant_contexts = list(), list(), list()
        ignore_substrings = " ".join([sentence[start_idx:end_idx] for [start_idx, end_idx, type_ent] in entity_dict['entities'] if type_ent in ignore_ents])
        doc = nlp(sentence)
        #print("sentence:", sentence)
        #print("ignoreing substrings:", ignore_substrings)
        #print("found nouns:")
        for j, token in enumerate(doc):
            if token.tag_.startswith("N") and not token.text in ignore_substrings:#noun which is not ignored
                #print("token:", token.text)
                #print("token dep type:", token.dep_)
                #print("token head text:", token.head.text)
                #print("token head pos", token.head.pos_)
                #print("children:", [child for child in token.children])
                relevant_tokens.append(token.text)
                window_left = min(0, j-2)
                window_right = min(j+3, len(doc))
                relevant_contexts.append(" ".join([token.text for token in doc[window_left:window_right]]))
        ret.append([ids[i], sentence, relevant_tokens, relevant_contexts])
    with open("output/untagged_nouns.pkl", "wb") as outf:
        pickle.dump(ret, outf)
    return ret


def _clean_and_tokenize(sentence):
    tokens = [re.sub('[^0-9a-zA-ZäöüÄÖÜ\s]+', '', w.text) for w in sentence]
    return tokens


def retrieve_main_description(data_corpus, untagged_nouns_list, n):

    #get all alphanumeric tokens and match to orginal id
    descriptions = [(re.sub('[^0-9a-zA-ZäöüÄÖÜ\s]+', ' ', description  ), idx) for idx, description in zip(data_corpus['index'], data_corpus['description'])]
    #descriptions = [(re.sub('[^0-9a-zA-ZäöüÄÖÜ\s]+', ' ', description  ), idx) for idx, description in zip(data_corpus['index'], data_corpus['dataset_title'])]

    desc_to_idx = dict(descriptions)
    idx_to_desc = {val:key for key,val in desc_to_idx.items()}
    descriptions = list(desc_to_idx.keys())
    tokenized_descriptions = []
    for doc in tqdm(nlp.pipe(descriptions, disable=["tagger", "parser","ner"])):
        tokens_doc = [token.text for token in doc]
        tokenized_descriptions.append(tokens_doc)
    #bm25 = BM25Okapi(tokenized_descriptions)
    #bm25 = BM25L(tokenized_descriptions)
    bm25 = BM25Plus(tokenized_descriptions)

    #counters for accuracy
    ttl, corr_single, corr_context, corr_query = 0, 0, 0, 0
    for [id_, sentence, relevant_tokens, relevant_contexts] in tqdm(untagged_nouns_list):
        ttl += 1
        #print("id:", id_)
        #print("goal desc:", idx_to_desc[id_])
        #print("sentence:", sentence)
    
        # first retrieve descriptions w/ bm25 for entire query
        top_docs_query = bm25.get_top_n(sentence  .split(), descriptions, n)
        ids_top_docs = [desc_to_idx[top] for top in top_docs_query]
        if id_ in ids_top_docs:
            corr_query += 1

        # second retrieve descriptions w/ bm25 for single tokens from query
        ids_top_docs_token = list()
        for relevant_token in relevant_tokens:
            top_docs_token = bm25.get_top_n([relevant_token  ], descriptions, n)
            ids_top_docs_token += [desc_to_idx[top] for top in top_docs_token]
        if id_ in ids_top_docs_token:
            corr_single += 1

        # third retrieve descriptions w/ bm25 for context tokens from query
        ids_top_docs_context = list()
        for relevant_context in relevant_contexts:
            top_docs_context = bm25.get_top_n(relevant_context  .split(), descriptions, n)
            ids_top_docs_context += [desc_to_idx[top] for top in top_docs_context]
        if id_ in ids_top_docs_context:
            corr_context += 1 

    print("Results for top {} bm25 retrieval:".format(n))
    print("Entire query accuracy: {:.2f}".format(corr_query/ttl*100.0))
    print("Relevant token accuracy: {:.2f}".format(corr_single/ttl*100.0))
    print("Relevant context accuracy: {:.2f}".format(corr_context/ttl*100.0))
    print()


if __name__ == "__main__":
    #Idee: Klassifier trainieren, der anhand der dep parses und POS tag Infos "main"-Vokabular vorhersagt -> oft ROOT

    tagged_dummy = load_training_data('input/tagged_sentences_latest.json')
    retrieval_data = load_retrieval_data('input/info_retrieval_data_latest.csv')
    overview_data = load_retrieval_data('data/datasets_overview.csv')

    relevant_sent2nouns = _find_entities(tagged_dummy, retrieval_data)

    for n in [10, 5, 3, 1]:
        retrieve_main_description(overview_data, relevant_sent2nouns, n)
        print('-------------------------')
    

