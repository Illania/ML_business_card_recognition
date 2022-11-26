import re

import nltk
import spacy

import locationtagger
 
# essential entity models downloads
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_addresses(text):
    place_entity = locationtagger.find_locations(text = text)
    return place_entity

#ent = get_addresses("555 Bryant, #106, Palo Alto, CA 94301")
#country = ent.countries