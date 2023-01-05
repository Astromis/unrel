from typing import List

import nltk
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.probability import FreqDist
from wordcloud import WordCloud

from .wrappers import get_gensim_lda


def get_freq_plot(tokens: List[str], title, top_n=50, fig_size=(15,5)):
    text = nltk.Text(tokens)
    fdist = FreqDist(text)
    plt.figure(figsize=fig_size)
    fdist.plot(top_n, cumulative=False, title=title)
    
def plot_token_diff(tokens_1: List[str], tokens_2: List[str], title, 
                    top_n=50, fig_size=(15,5)):
    unique_bigrams = set(tokens_1) - set(tokens_2)
    unique_fdict = FreqDist()
    tf = FreqDist(tokens_1)
    for t in unique_bigrams:
        unique_fdict[t] = tf[t]
    plt.figure(figsize=fig_size)
    unique_fdict.plot(top_n,cumulative=False, title=title)

def get_hist(df, x:str, hue:str):
    sns.histplot(data=df, x=x, hue=hue, kde=True)

def entity_bar_plot(entites: List[str], top_n=10, figsize=(10,5)):
    s = pd.Series(entites)
    plt.figure(figsize=figsize)
    s.value_counts().iloc[:top_n].plot.bar()
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

""" get_freq_plot(
            tokenize_from_list_strings(negative_text, russian_stopwords), 
            "take_off_tokens")
            
plot_token_diff(
                tokenize_from_list_strings(positive_text, russian_stopwords, ngram=1),
                tokenize_from_list_strings(negative_text, russian_stopwords, ngram=1),
                title="take_off_unique_bigrams") """

def get_word_cloud(text: str, stopwords=None, max_words=100, max_font_size=30,
                    scale=3, random_state=1, figsize=(12,12)):

    wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=max_words,
            max_font_size=max_font_size, 
            scale=scale,
            random_state=random_state)
            
    wordcloud = wordcloud.generate(text)

    # wordcloud.to_svg("wordcloud.svg")
    fig = plt.figure(1, figsize=figsize)
    plt.axis('off')
    
    plt.imshow(wordcloud)
    plt.show()

def plot_lda_vis(corpus: List[List[str]], num_topics):
    """https://app.neptune.ai/neptune-ai/eda-nlp-tools/n/9-0-eda-for-nlp-template-e8339407-5fc0-47cf-a304-994c0e2a689e/a9606b25-9906-439b-b141-165a162a874d"""
    lda_model, dic, bow_corpus = get_gensim_lda(corpus, num_topics, supplumentary=True)
    #lda_model.show_topics()
    pyLDAvis.enable_notebook()   
    vis = gensimvis.prepare(lda_model, bow_corpus, dic)
    return vis
