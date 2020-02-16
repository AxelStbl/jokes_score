import os, re
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import os
import re
import requests
from collections import Counter
import itertools

from spacy.tokenizer import Tokenizer
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import featuretools as ft
import featuretools.variable_types as vtypes
from featuretools.primitives import AggregationPrimitive

import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from nltk.corpus import brown
import re
import zipfile
import requests
from bs4 import BeautifulSoup
from collections import Counter
import itertools

import spacy
from spacy.tokenizer import Tokenizer
import gensim as gen
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import featuretools as ft
import featuretools.variable_types as vtypes
from featuretools.primitives import AggregationPrimitive
from nlp_primitives import (
    DiversityScore,
    LSA,
    MeanCharactersPerWord,
    PolarityScore,
    UniversalSentenceEncoder,
    PunctuationCount,
    StopwordCount,
    TitleWordCount,
    UpperCaseCount)


def eda():
    if os.path.exists("./eda_reddit_jokes.pkl"):
        with open("./eda_reddit_jokes.pkl", "r") as f:
            return pd.read_pickle('./eda_reddit_jokes.pkl')
    df = pd.read_json(
        'redit_jokes.json')
    df = df[df.score > 0].reset_index()
    df['joke'] = df['title'] + '. ' + df['body']
    df = df.loc[:, ['joke', 'score']]
    df.to_pickle(
        './eda_reddit_jokes.pkl')

    return df


def feature_engineering():
    if features_files_exist():
        return pd.read_pickle("./X_train.pkl"), pd.read_pickle(
            "./y_train.pkl"), pd.read_pickle("./X_test.pkl"), pd.read_pickle(
            "./y_test.pkl")
    else:
        df = eda().iloc[0:500,:]
        df['joke_words'] = df['joke'].apply(lambda x: len(x.split(' ')))
        df = df[df['joke_words'] <= 40].reset_index()
        del df['index']
        del df['joke_words']

        ambigous_words(df)

        nlp = spacy.load('en')

        token_pos = joke_tokenized(df, nlp)
        best_score_similarity_words(df, token_pos)
        antonyms(df, token_pos)
        longest_word(df, token_pos)
        speical_chars(df, token_pos)

        feature_matrix_customers, features_defs = create_scores(df)
        df = pd.concat([df['joke'], feature_matrix_customers], axis=1)

        df = object_count_column(df)

        df = add_pos_count_columns(df)

        ## Text pre-processing
        text_preprocessing(df, nlp)
        total_words_chars(df)

        target = 'score'
        X = df[df.columns[df.columns != target]]
        y = df[target]

        ## Use validation set
        # Split earlier to avoid leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)
        X_test, X_train, y_test, y_train = vectorise(X_test, X_train, y_test,
                                                     y_train)

        pd.to_pickle(X_train, "X_train.pkl")
        pd.to_pickle(X_test, "X_test.pkl")
        pd.to_pickle(y_train, "y_train.pkl")
        pd.to_pickle(y_test, "y_test.pkl")
    return X_train, y_train, X_test, y_test


def vectorise(X_test, X_train, y_test, y_train):
    X_train.reset_index(inplace=True)
    X_test.reset_index(inplace=True)
    y_train = y_train.reset_index()['score']
    y_test = y_test.reset_index()['score']
    X_train = X_train.loc[:, X_train.columns != 'index']
    X_test = X_test.loc[:, X_test.columns != 'index']
    vectorizer = TfidfVectorizer(input='content', decode_error='strict',
                                 lowercase=True, stop_words='english',
                                 ngram_range=(1, 2),
                                 max_features=1000, binary=True)
    vectorizer.fit(X_train['joke_processed_tokenized_stemmed_str'])
    tfidf_vec_train = vectorizer.transform(
        X_train['joke_processed_tokenized_stemmed_str']).toarray()
    tfidf_vec_test = vectorizer.transform(
        X_test['joke_processed_tokenized_stemmed_str']).toarray()
    X_train = X_train.merge(
        pd.DataFrame(tfidf_vec_train, columns=vectorizer.vocabulary_),
        how='left', left_index=True, right_index=True)
    X_test = X_test.merge(
        pd.DataFrame(tfidf_vec_test, columns=vectorizer.vocabulary_),
        how='left', left_index=True, right_index=True)
    return X_test, X_train, y_test, y_train


def total_words_chars(df):
    df['total_words'] = df['joke_tokenized'].apply(len)
    df['total_chars'] = df['joke'].apply(len)


def text_preprocessing(df, nlp):
    def replace_non_eng_punct(txt):
        return re.sub(r'/[^a-zA-Z0-9\s,.?!]/', '*', txt).strip()

    def replace_escape(txt):
        updated_txt = re.sub(r'\n|\t|&amp;', ' ', txt)
        return updated_txt.strip()

    def remove_multi_spaces(txt):
        return re.sub(' +', ' ', txt)

    def preprocess_text(document):
        #         # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        #         # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        #         # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        return document

    df['joke_text_processed'] = df['joke'].apply(
        replace_non_eng_punct).apply(
        remove_multi_spaces).apply(replace_escape)
    df['joke_text_processed'] = df['joke_text_processed'].apply(
        preprocess_text)
    df['joke_processed_tokenized'] = df['joke_text_processed'].apply(nlp)
    stemmer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    lemmatize_remove_stop_words = lambda x: nlp(' '.join(
        [stemmer.lemmatize(str(word)) for word in x if
         str(word) not in stop_words]))
    lemmatize_remove_stop_words_str = lambda x: ' '.join(
        [stemmer.lemmatize(str(word)) for word in x if
         str(word) not in stop_words])
    df['joke_processed_tokenized_stemmed'] = df[
        'joke_processed_tokenized'].apply(lemmatize_remove_stop_words)
    df['joke_processed_tokenized_stemmed_str'] = df[
        'joke_processed_tokenized'].apply(lemmatize_remove_stop_words_str)


def add_pos_count_columns(df):
    def add_pos_count_cols(df, tokenized_col):
        df['temp'] = df[tokenized_col].apply(
            lambda x: [ent.pos_ for ent in x])
        mlb = MultiLabelBinarizer()
        mlb.fit(df['temp'])
        df = df.join(pd.DataFrame(mlb.transform(df['temp']),
                                  columns=mlb.classes_,
                                  index=df.index))
        del df['temp']
        return df

    df = add_pos_count_cols(df, 'joke_tokenized')
    return df


def object_count_column(df):
    def add_object_count_cols(df, tokenized_col):
        df['temp'] = df[tokenized_col].apply(
            lambda x: [ent.label_ for ent in x.ents])
        mlb = MultiLabelBinarizer()
        mlb.fit(df['temp'])
        df = df.join(pd.DataFrame(mlb.transform(df['temp']),
                                  columns=mlb.classes_,
                                  index=df.index))
        del df['temp']
        return df

    df['joke_tokenized'].apply(lambda x: [ent.label_ for ent in x.ents])
    df = add_object_count_cols(df, 'joke_tokenized')
    return df


def create_scores(df):
    es = ft.EntitySet("jokes_df")
    es.entity_from_dataframe(entity_id="jokes_df",
                             index="joke_id",
                             make_index=True,
                             dataframe=df)
    """
        DiversityScore()
        Calculates the overall complexity of the text based on the total

        LSA()
        Calculates the Latent Semantic Analysis Values of Text Input

        MeanCharactersPerWord()
        Determines the mean number of characters per word.

        PolarityScore()
        Calculates the polarity of a text on a scale from -1 (negative) to 1 (positive)

        PunctuationCount()
        Determines number of punctuation characters in a string.

        StopwordCount()
        Determines number of stopwords in a string.

        TitleWordCount()
        Determines the number of title words in a string.

        UpperCaseCount()
        Calculates the number of upper case letters in text.
        """
    trans = [
        #     DiversityScore,
        #          LSA,
        MeanCharactersPerWord,
        UniversalSentenceEncoder,
        PolarityScore,
        PunctuationCount,
        StopwordCount,
        TitleWordCount,
        UpperCaseCount]
    feature_matrix_customers, features_defs = ft.dfs(entityset=es,
                                                     target_entity='jokes_df',
                                                     #   instance_ids=["joke"],
                                                     trans_primitives=trans,
                                                     max_depth=4)
    return feature_matrix_customers, features_defs


def speical_chars(df, token_pos):
    def count_speical_chars(tokens_poss):
        """ how many speical chars in string """
        bin_isalphanumberic_list = [not word_pos[0].isalnum() for word_pos
                                    in tokens_poss]

        return np.sum(bin_isalphanumberic_list)

    df['speical_chars'] = token_pos.apply(count_speical_chars)


def longest_word(df, token_pos):
    def len_longest_word(tokens_poss):
        """ check the length of the longest word """
        word_len_list = [(word_pos[0], len(word_pos[0])) for word_pos in
                         tokens_poss]
        word_len_list = sorted(word_len_list, key=lambda x: x[1])

        return word_len_list[-1][1]

    df['longest_word'] = token_pos.apply(len_longest_word)


def antonyms(df, token_pos):
    list_of_antonyms = []
    from nltk.corpus import wordnet as wn
    for i in wn.all_synsets():
        if i.pos() in ['a', 's']:
            for j in i.lemmas():
                if j.antonyms():
                    (j.name(),
                     j.antonyms()[0].name()) and list_of_antonyms.append(
                        (j.name(), j.antonyms()[0].name()))
    dict_antonyms = dict((y, x) for x, y in list_of_antonyms)

    def find_antonyms(tokens_poss):
        """ count how many antonyms in sentence """
        word_list = [word_pos[0] for word_pos in tokens_poss]
        count = 0
        for word in word_list:
            try:
                if word in dict_antonyms:
                    if dict_antonyms[word] in word_list:
                        count += 1
            except Exception:
                pass  # don't found in dict
        return count

    df['antonyms'] = token_pos.apply(find_antonyms)


def best_score_similarity_words(df, token_pos):
    nltk.download('brown')
    b = Word2Vec(brown.sents())

    def find_best_similarity(string):
        """ find the max similarity between two words """
        # unique words:
        word_list = list(set(
            [word_pos[0] for word_pos in string if
             word_pos[1] != "PROPN"]))
        all_combi = list(itertools.combinations(word_list, 2))

        max_similarity = 0
        for pairs in all_combi:
            try:
                temp_similatity = b.similarity(pairs[0], pairs[1])
                if temp_similatity > max_similarity:
                    max_similarity = temp_similatity
            except:
                pass  # don't found the word in word2vec
        return max_similarity

    df['best_score_similarity_words'] = token_pos.apply(
        find_best_similarity)


def joke_tokenized(df, nlp):
    df['joke_tokenized'] = df['joke'].apply(nlp)
    token_pos = df['joke_tokenized'].apply(
        lambda x: [(elm.text, elm.pos_) for elm in x])
    return token_pos


def ambigous_words(df):
    def get_ambiguous_words():
        res = requests.get('https://muse.dillfrog.com/lists/ambiguous')
        page_soup = BeautifulSoup(res.content, features="lxml")
        a_tags = page_soup.find_all('a',
                                    href=re.compile(r'.*/meaning/word/*'))
        return [word.text for word in a_tags]

    ambiguous_words = get_ambiguous_words()
    df['ambiguous_words'] = df['joke'].apply(
        lambda x: sum([str(w) in ambiguous_words for w in x]))


def features_files_exist():
    features_files = ["./X_train.pkl", "./X_test.pkl", "./y_train.pkl",
                      "./y_test.pkl"]
    file_exists = True
    for file in features_files:
        if not os.path.exists(file):
            return False
    return file_exists

if __name__ == '__main__':
    print(feature_engineering()[0].head())
