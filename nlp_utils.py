


# %% -------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
# %% -------------------------------------------------------------------------
# Objects that don't change between object calls
# ----------------------------------------------------------------------------
# Stop words
stopword_list = stopwords.words('english')
# Word dictionary 
word_dict = set(word.lower() for word in nltk.corpus.words.words())
# Lemmatiser
lemmatizer = WordNetLemmatizer()
# %% -------------------------------------------------------------------------
# Utility function
# ----------------------------------------------------------------------------
def case_normalize(text):
    '''
    Takes in a text string and returns a string with all characters in lower case. 
    '''
    text = text.lower()
    return text
 
def remove_punctuation(text):
    '''
    Takes in a text string and returns a string with all the punctuation removed.
    '''
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text
 
def tokenize(text):
    '''
    Takes in a text string and returns a list where each item corresponds to a token.
    '''
    tokenized_text = word_tokenize(text)
    return tokenized_text
 
def remove_stopwords(tokenized_text):
    '''
    Takes in a list of text, and returns another list where the stop words have been removed.
    '''
    processed_text = [word for word in tokenized_text if word not in stopword_list]
    return processed_text
 
def remove_unknown_words(tokenized_text):
    '''
    Takes in a list of text, and returns another list where unknown words have been removed.
    '''
    processed_text = [word for word in tokenized_text if word in word_dict]
    return processed_text
 
def lemmatize(tokenized_text):
    '''
    Takes in a list of text, and returns another list where each word has been lemmatized.
    '''
    sample_doc_lemmatised = [lemmatizer.lemmatize(word) for word in tokenized_text]
    return sample_doc_lemmatised
 
def process_text(text):
    '''
    Takes in a raw text document and performs the following steps in order:
    - punctuation removal
    - case normalization
    - tokenization
    - remove stopwords
    - lemmatization
 
    Then returns a string containing the processed text
    '''
    processed_text = case_normalize(text)
    processed_text = remove_punctuation(processed_text)
    processed_text = tokenize(processed_text)
    processed_text = remove_stopwords(processed_text)
    processed_text = lemmatize(processed_text)
    processed_text = ' '.join(processed_text)
    return processed_text