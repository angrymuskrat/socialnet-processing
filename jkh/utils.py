import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('wordnet')

eng_stopwords = stopwords.words('english')

from nltk.stem import WordNetLemmatizer

from gensim.models import fasttext as ft, Word2Vec
from gensim.test.utils import datapath

import fasttext

from cleantext import clean

from transformers import (
    AutoModel, 
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import pipeline

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.callbacks import ModelCheckpoint 

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import pandas as pd
import numpy as np
from scipy import sparse
from tqdm.notebook import tqdm

import re
from itertools import chain, islice
import logging
import os
from collections import Counter
import time

CORES = 10

SEED = 42
TRAIN_DOC_COUNT = 10000
TEST_DOC_COUNT = 1000
AUTHOR_COUNT = 100


def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def remove_mentions(data):
    return re.sub(r'(@[\w_]+)|(\[.*\](,? )?)', '', data)

def split_hashtags(data):
    return re.sub(r'#', ' #', data)

def remove_extra_symbols(data):
    return re.sub(r'[\(\)_]+', '', data)

def remove_html_tags(data):
    return re.sub(r'<.*?>', ' ', data)

def remove_expand(data):
    return re.sub(r'expand text…', '', data)

def apply_clean(doc, vk_preprocess=False):
    doc = remove_html_tags(doc)
    doc = clean(doc,
                fix_unicode=True,               # fix various unicode errors
                to_ascii=False,                  # transliterate to closest ASCII representation
                lower=True,                     # lowercase text
                no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
                no_urls=True,                  # replace all URLs with a special token
                no_emails=True,                # replace all email addresses with a special token
                no_phone_numbers=True,         # replace all phone numbers with a special token
                no_numbers=False,               # replace all numbers with a special token
                no_digits=False,                # replace all digits with a special token
                no_currency_symbols=True,      # replace all currency symbols with a special token
                no_punct=False,                 # remove punctuations
                replace_with_punct="",          # instead of removing punctuations you may replace them
                replace_with_url="",
                replace_with_email="",
                replace_with_phone_number="",
                replace_with_currency_symbol="",
                lang="en"                       # set to 'de' for German special handling
               )
    doc = remove_emojis(doc)
    if vk_preprocess:
        doc = remove_mentions(doc)
        doc = split_hashtags(doc)
        doc = remove_extra_symbols(doc)
        doc = remove_expand(doc)
    
    return doc


def get_text_and_hashtags(row, text_col='text'):
    text = row[text_col]
    if len(text) == 0:
        return ''
    return re.sub('#\w+', '', text).strip(), ' '.join(re.findall('#\w+', text))