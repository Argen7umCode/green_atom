import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.utils import pad_sequences
from collections import Counter
from pandarallel import pandarallel
from pprint import pprint
from re import sub





class DataPreProcesser:

    def __init__(self, max_lenth) -> None:
        nltk.download('stopwords')
        self.max_lenth = max_lenth
        self.stopwords = None
        self.lemmatizer = WordNetLemmatizer()
        
    
    def set_stopwords(self, stopwords):
        self.stopwords = stopwords

    def set_word2vec_matrix(self, word2vec_matrix):
        self.word2vec_matrix = word2vec_matrix

    def set_rex_to_rem_stopwords(self):
        if not self.stopwords:
            nltk.download('stopwords')
            self.stopwords = list(stopwords.words('english'))
        
        self.rex_to_rem_stopwords = '|'.join(f'\s{word}\s' for word in self.stopwords)

    def clean_regex(self, text):
        text = sub('\"{2,3}', '', text)
        text = sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)
        text = sub(r'[.,!?-]', '', text)
        text = sub(r'<[^>]+>', '', text)
        text = sub(r'\d+', '', text)
        text = sub(r' +', ' ', text)
        text = sub(r'[^\x00-\x7f]', r'', text)
        return text

    def tokenize_text(self, text):
        return word_tokenize(text)

    def split_tords_by_space(self, text):
        return text.split()

    def make_lower(self, word_list):
        return list(map(lambda t: t.lower(), word_list))

    def lemmatize(self, word_list):
        return list(map(self.lemmatizer.lemmatize, word_list))

    def remove_stop_words(self, word_list):
        text = self.join_word_list(word_list)
        text = sub(self.rex_to_rem_stopwords, ' ', text)
        return self.split_tords_by_space(text)
    
    def add_bos_tag(self, word_list):
        word_list.insert(0, '<bos>')
        return word_list

    def limit_sequence(self, word_list, max_lenth):
        return word_list[:max_lenth]

    def join_word_list(self, word_list):
        return ' '.join(word_list)

    def pad_seq(self, sequence):
        print(sequence)
        return list(pad_sequences([sequence], self.max_lenth, padding='pre', value=0))

    def preprocess_text(self, text):
        word_list = self.tokenize_text(self.clean_regex(text))
        lower_word_list = self.make_lower(word_list)
        lemmatized_word_list = self.make_lower(lower_word_list)
        word_list_with_bos = self.add_bos_tag(self.remove_stop_words(lemmatized_word_list))
        preprocessed_seq = self.limit_sequence(word_list_with_bos, self.max_lenth)

        return preprocessed_seq


