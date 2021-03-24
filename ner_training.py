import spacy
import warnings
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

# helper function for incrementing the revision counters
def increment_revision_counters(entity_counter, entities):
    for entity in entities:
        label = entity[2]
        if label in entity_counter:
            entity_counter[label] += 1
        else:
            entity_counter[label] = 1

def load_training_data(file):
    with open(file) as json_file:
        sentences = json.load(json_file)
    print("LENGTH OF DATASET: ", len(sentences))
    
    dataset_dict, entity_counter = dict(), dict()
    for sentence in sentences:
        entities = sentence[1]["entities"]
        increment_revision_counters(dataset_dict, entities)
        # for entity in entities:
        #     label = entity[2]
        #     try:
        #         entity_counter[label] += 1
        #     except KeyError:
        #         entity_counter[label] = 1
    pprint(entity_counter)
    return sentences

def load_retrieval_data(file):
    with open(file, 'r') as csv_file:
        data = pd.read_csv(csv_file)
    return data
    
def load_external_sentences(file):
    npr_df = pd.read_csv(file, delimiter = "\t")
    npr_df = npr_df.sample(frac=1)
    return npr_df

def prepare_revision_data(sentence_count=10000, batch_size=50):
    npr_df = load_external_sentences("external/deu_news_2015_3M-sentences.txt")

    revision_texts = []


    # convert the articles to spacy objects to better identify the sentences. Disabled unneeded components. # takes ~ 4 minutes
    for doc in tqdm(nlp.pipe(npr_df.iloc[:sentence_count,1], batch_size=batch_size, disable=["tagger", "ner"])):
        for sentence in doc.sents:

            if  40 < len(sentence.text) < 80:
                # some of the sentences had excessive whitespace in between words, so we're trimming that
                revision_texts.append(" ".join(re.split("\s+", sentence.text, flags=re.UNICODE)))  

    revisions = []

    # Use the existing spaCy model to predict the entities, then append them to revision
    for doc in nlp.pipe(revision_texts, batch_size=batch_size, disable=["tagger", "parser"]):
        
        # don't append sentences that have no entities
        if len(doc.ents) > 0:
            revisions.append((doc.text, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}))

    return revisions


def split_revision_data(REVISION_SENTENCE_SOFT_LIMIT = 100):
    # create arrays to store the revision data
    TRAIN_REVISION_DATA = []
    TEST_REVISION_DATA = []

    # create dictionaries to keep count of the different entities
    TRAIN_ENTITY_COUNTER = {}
    TEST_ENTITY_COUNTER = {}

    random.shuffle(revisions)
    for revision in revisions:
        # get the entities from the revision sentence
        entities = revision[1]["entities"]

        # simple hack to make sure spaCy entities don't get too one-sided
        should_append_to_train_counter = 0
        for _, _, label in entities:
            if label in TRAIN_ENTITY_COUNTER and TRAIN_ENTITY_COUNTER[label] > REVISION_SENTENCE_SOFT_LIMIT:
                should_append_to_train_counter -= 1
            else:
                should_append_to_train_counter += 1

        # simple switch for deciding whether to append to train data or test data
        if should_append_to_train_counter >= 0:
            TRAIN_REVISION_DATA.append(revision)
            increment_revision_counters(TRAIN_ENTITY_COUNTER, entities)
        else:
            TEST_REVISION_DATA.append(revision)
            increment_revision_counters(TEST_ENTITY_COUNTER, entities)

    return TRAIN_REVISION_DATA, TEST_REVISION_DATA


def create_train_test_set(sentences, TRAIN_REVISION_DATA):
    random.shuffle(sentences)
    TRAIN_STAT_DATA = sentences[:int(len(sentences)*0.8)]
    TEST_STAT_DATA = sentences[int(len(sentences)*0.8):]

    print(len(sentences))
    print(len(TRAIN_STAT_DATA))
    print(len(TEST_STAT_DATA))
    print("REVISION", len(TRAIN_REVISION_DATA))
    TRAIN_DATA = TRAIN_REVISION_DATA + TRAIN_STAT_DATA
    print("COMBINED", len(TRAIN_DATA))

    return TRAIN_DATA, TEST_STAT_DATA

def train_ner(TRAIN_DATA, epochs=30):
    #STAT: below is the heart piece of this script, and the code was heavily changed compared to the original
    #script taken out of the code on deepnote.com. The reason is thaat this code has been adapted to spacy 3 -
    #while the old code was running on spacy 2.X
    #central command is nlp-update

    ner = nlp.get_pipe("ner")

    ner.add_label("GRAN")
    ner.add_label("PLACE")
    ner.add_label("TIME")

    # get the names of the components we want to disable during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # start the training loop, only training NER
    #optimizer = nlp.resume_training()
    #optimizer = nlp.initialize()
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        sizes = compounding(1.0, 4.0, 1.001)
        
        # batch up the examples using spaCy's minibatc
        for epoch in range(epochs):
            random.shuffle(TRAIN_DATA)
            #text = []
            #annots=[]
            examples=[]


            for text,annots in TRAIN_DATA:
                #text.append(t)
                #annots.append(a)
                doc = nlp.make_doc(text)    
                example = Example.from_dict(doc, annots)
                examples.append(example)
            
            losses = {}
            
            nlp.update(examples, drop=0.35, losses=losses)#,sgd=optimizer)

            print("Losses ({}/{})".format(epoch + 1, epochs), losses)


############################### EVALUATION ###############################

def display_sentences():
    statbot_colors = {"GRAN": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
                  "PLACE": "linear-gradient(90deg, #ffff00, #ff8c00)",
                  "TIME": "linear-gradient(90deg, #ba9cfc, #ac9ce7)"}
    statbot_options = {"ents": ["PER","LOC","ORG","MISC","GRAN","PLACE", "TIME"], "colors": statbot_colors}
    spacy.displacy.render(nlp("Ich heisse Christian und war heute in Zürich bei IBM im Internet."), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Wie viele Kühe hat die Gemeinde Bülach?"), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Wie hoch ist Eigenkapital auf Bezirksebene?"), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Ich brauche die Daten pro Bezirk"), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Ich brauche die Daten für den gesamten Kanton."), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Wie viel Bauinv. EFH 5 Jahre  hat  in Regensdorf  ?"), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Was ist der Anteil an MIV-Anteil (Modal Split)   auf Bezirksebene ?"), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Was ist der Anteil an Geb.Vol. Dienstleistungen: Zunahme   in Flaach ?"), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Welches ist das Schül. Sekundarstufe II   für den gesamten Kanton ?"), style="ent",options=statbot_options)
    spacy.displacy.render(nlp("Welche Gemeinde hat die grösste Bevölkerung?"), style="ent",options=statbot_options)


def evaluate(TEST_STAT_DATA):
    # dictionary to hold our evaluation data
    stat_evaluation = {
        "GRAN": {
            "correct": 0,
            "total": 0,
        },
        "PLACE": {
            "correct": 0,
            "total": 0,
        },
        "TIME": {
            "correct": 0,
            "total": 0,
        }
    }


    for stat in TEST_STAT_DATA:
        # extract the sentence and correct stat entities according to our test data
        sentence = stat[0]
        entities = stat[1]["entities"]

        # for each entity, use our updated model to make a prediction on the sentence
        for entity in entities:
            doc = nlp(sentence)
            correct_text = sentence[entity[0]:entity[1]]
            
            # if we find that there's a match for predicted entity and predicted text, increment correct counters
            for ent in doc.ents:
                print("ENT_LABEL",ent.label_)
                print("ENTITY2",entity[2])
                print("ENT_TEXT",ent.text)
                print("CORRECT:TEXT",correct_text)
                if ent.label_ == entity[2] and ent.text == correct_text:
                    
                    stat_evaluation[entity[2]]["correct"] += 1
                    
                    # this break is important, ensures that we're not double counting on a correct match
                    break

            #  increment total counters after each entity loop
            stat_evaluation[entity[2]]["total"] += 1

    stat_total_sum = 0
    stat_correct_sum = 0

    for key in stat_evaluation:
        correct = stat_evaluation[key]["correct"]
        total = stat_evaluation[key]["total"]
        
        stat_total_sum += total
        stat_correct_sum += correct

        print(f"{key}: {correct / total * 100:.2f}%")

    print(f"\nTotal: {stat_correct_sum/stat_total_sum * 100:.2f}%")


# helper function to udpate the entity_evaluation dictionary
def update_results(entity, metric, entity_evaluation):
    if entity not in entity_evaluation:
        entity_evaluation[entity] = {"correct": 0, "total": 0}
        
    entity_evaluation[entity][metric] += 1

def evaluate_existing_entities(TEST_REVISION_DATA):
    # dictionary which will be populated with the entities and result information
    entity_evaluation = {}

    # same as before, see if entities from test set match what spaCy currently predicts
    for data in TEST_REVISION_DATA:
        sentence = data[0]
        entities = data[1]["entities"]

        for entity in entities:
            doc = nlp(sentence)
            correct_text = sentence[entity[0]:entity[1]]

            for ent in doc.ents:
                if ent.label_ == entity[2] and ent.text == correct_text:
                    update_results(ent.label_, "correct", entity_evaluation)
                    break

            update_results(entity[2], "total", entity_evaluation)

    sum_total = 0
    sum_correct = 0

    for entity in entity_evaluation:
        total = entity_evaluation[entity]["total"]
        correct = entity_evaluation[entity]["correct"]

        sum_total += total
        sum_correct += correct
        
        print("{} | {:.2f}%".format(entity, correct / total * 100))

    print()
    print("Overall accuracy: {:.2f}%".format(sum_correct / sum_total * 100))        


if __name__ == "__main__":
    dataset = load_training_data('input/tagged_sentences_latest.json')
    pprint(dataset[:10])

    retrieval_data = load_retrieval_data('input/info_retrieval_data_latest.csv')

    revisions = prepare_revision_data(sentence_count=10000, batch_size=50)

    # print an example of the revision sentence
    #print(revisions[0][0])

    # print an example of the revision data
    #print(revisions[0][1])

    TRAIN_REVISION_DATA, TEST_REVISION_DATA = split_revision_data()

    TRAIN_DATA, TEST_STAT_DATA = create_train_test_set(dataset, TRAIN_REVISION_DATA)

    train_ner(TRAIN_DATA=TRAIN_DATA, epochs=30)

    #display_sentences()

    evaluate(TEST_STAT_DATA)

    evaluate_existing_entities(TEST_REVISION_DATA)