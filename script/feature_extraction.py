#-- Import -------------------------------------------------------------------------------------
import pandas as pd
import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, RSLPStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
#-----------------------------------------------------------------------------------------------

#-- Download NLTK ------------------------------------------------------------------------------
nltk.download('punkt')
nltk.download('stopwords')
#-----------------------------------------------------------------------------------------------

#-- Initialize ---------------------------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#-----------------------------------------------------------------------------------------------

#-- Function to Remove Stopwords ---------------------------------------------------------------
def remove_stopwords(text, language='english', domain_stopwords=None):
    if domain_stopwords is None:
        domain_stopwords = []

    stop_words = set(stopwords.words(language))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)

    filtered_tokens = [token for token in tokens if token not in stop_words
                       and token not in domain_stopwords
                       and not token.isdigit()
                       and len(token) > 1]

    return ' '.join(filtered_tokens)
#-----------------------------------------------------------------------------------------------

#-- Function to Steaming ---------------------------------------------------------------
def apply_stemming(text, language='english'):
    stemmer = PorterStemmer() if language == 'english' else RSLPStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)
#-----------------------------------------------------------------------------------------------

#-- Function to Preprocess text content of news -----------------------------------------------
def preprocess_text_column(df, text_column='txt', language='english', domain_stopwords=None):
    processed_texts = []
    for text in df[text_column]:
        text = remove_stopwords(text, language=language, domain_stopwords=domain_stopwords)
        text = apply_stemming(text, language=language)
        processed_texts.append(text)

    df[text_column] = processed_texts
    return df
#-----------------------------------------------------------------------------------------------

#-- Function to Create embeddings for total rows in df -----------------------------------------
def generate_embeddings_using_doc2vec(ds_name, ds_lang):

    #-- log --
    print('Creating Text Embeddings using Doc2Vec Model ...')

    #-- load df --
    data_dir = os.path.join(base_dir, 'data/' + ds_name)
    df_file = os.path.join(data_dir, ds_name + '.csv')
    df = pd.read_csv(df_file)

    #-- preprocess --
    print('Preprocessing ...')
    df = preprocess_text_column(df, text_column='txt', language=ds_lang)
    print(df.shape)

    #-- create model --
    print('Initializing doc2vec model ...')
    dim_size = 500
    max_epochs = 100
    window_size = 8
    num_threads = 4
    min_count = 1
    alpha = 0.025
    min_alpha = 0.0001
    model = Doc2Vec(vector_size=dim_size,
                    alpha=alpha,
                    min_alpha=min_alpha,
                    min_count=min_count,
                    window=window_size,
                    workers=num_threads,
                    dm=1)

    #-- run fast-text --
    print('Creating embeddings ...')
    documents = [TaggedDocument(words=row.split(), tags=[str(i)]) for i, row in enumerate(df['txt'])]
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    number_of_data = len(df)
    number_of_features = model.vector_size
    X = np.zeros((number_of_data, number_of_features))
    for i in range(number_of_data):
        X[i] = model.dv[str(i)]

    vectorized_df = pd.DataFrame(X, columns=[f'txt_embd_{i}' for i in range(number_of_features)])
    df = pd.concat([df, vectorized_df], axis=1)

    print(f'Embeddings Created:\nds-size:{df.shape}')

    #-- save --
    print('Saving results ...')
    result_file = os.path.join(data_dir, ds_name + '_embeddings.csv')
    df.to_csv(result_file, index=False)

    print('create_FastText_embeddings: DONE :)\n')
#-----------------------------------------------------------------------------------------------